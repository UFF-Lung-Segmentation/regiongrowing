#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <cuda.h>

#define IMAGE_INPUT_DIR "dataset/Cx2_Ima1/csv"
#define IMAGE_OUTPUT_DIR "dataset/Cx2_Ima1/region"
#define WIDTH 512
#define MAX_NUMBER_CORTES 300

// Limiar area dos pulmões
#define HU_PULMAO_MIN -700
#define HU_PULMAO_MAX -600
      
// MIN/MAX das tomografias dos ratos
#define MIN_HU -1024
#define MAX_HU 100

// número de features extraídas de cada voxel
#define NUM_FEATURES 5
#define LIMIAR 0.9

// constantes para trabalhar com arqivos
#define MAX_LINE_SIZE 3072 // se pixel value 16bit: valores -32768 a +32767: 6 caracteres * 512 elementos por linha = 3.072 
#define MAX_TOKEN_SIZE 6
#define MAX_FILENAME 1024  


// *********************************************************************
// funcao que informa a quantidade de cortes no diretório
// *********************************************************************
int countSlices(){
  DIR *d;
  struct dirent *dir;
  d = opendir(IMAGE_INPUT_DIR);
  int num_slices = 0;
  if (d){
      while ((dir = readdir(d)) != NULL)
      {
        if (dir->d_type == DT_REG){
          num_slices++;
        }
      }
      closedir(d);
  } else {
    printf("countSlices: não conseguiu ler o diretório\n");
    return(-1);
  }
  return num_slices;
}

// *********************************************************************
// funcao para carregar os cortes do filesystem para a memória principal
// *********************************************************************
int loadCT(int *imagem){

  
  // verifica os arquivos no diretorio
  DIR *d;
  struct dirent *dir;
  d = opendir(IMAGE_INPUT_DIR);
  char files[MAX_NUMBER_CORTES][MAX_FILENAME];
  int num_files = 0;
  if (d){
      while ((dir = readdir(d)) != NULL)
      {
        if (dir->d_type == DT_REG){
          char filename[MAX_FILENAME] = IMAGE_INPUT_DIR "/";
          strcpy(files[num_files++], strcat(filename,dir->d_name));
        }
      }
      closedir(d);
  } else {
    printf("loadCT: não conseguiu ler o diretório\n");
    return(-1);
  }

  // ordena a lista de arquivos
  for (int i = 0; i < num_files; i++){
    for (int j = 0; j < num_files; j++){      
      if (strcmp(files[i], files[j]) < 0){
        char temp[MAX_FILENAME] = {};
        strcpy(temp,files[i]);
        strcpy(files[i], files[j]);
        strcpy(files[j],temp);
      }
    }
  } 

  // carrega cada corte na memória
  int ntoken = 0;
  for (int i = 0; i < num_files; i++){
    FILE *file = NULL;
    file = fopen(files[i], "r");
    if (!file){
      printf("loadCT: não conseguiu abrir arquivo\n");
      return(-2);
    }
    int nlines = 0;
    char *pbuf;
    char buf[MAX_LINE_SIZE] = {};
    while (pbuf = fgets(buf, sizeof(buf), file)){ // le a linha do arquivo
      char *p = pbuf;
      while ((p=strchr(pbuf, ',')) != NULL || (p=strchr(pbuf, '\n')) != NULL){  // obtem cada valor de pixel
        int len = p - pbuf;
        char token[MAX_TOKEN_SIZE];
        int k= 0;
        for (; k < len; k++){
          token[k] = pbuf[k];
        }
        token[k] = '\0';
        pbuf = p+1;
        imagem[ntoken++] = atoi(token);  
      }
      ++nlines;
    }
    fclose(file);      
  }  
  return(0);  
}

// *********************************************************************
// funcao para salvar os arquivos em disco
// *********************************************************************
int saveCT(int *imagem, int num_slices){
  
  int pixels_por_slice = WIDTH * WIDTH;
  
  char filename[MAX_FILENAME];
  char filepath[MAX_FILENAME];
  
  int cursor=0;
  
  for (int i = 0; i < num_slices; i++){
    snprintf(filename, 14, "/slice_%02d.txt", i);
    strcpy(filepath, IMAGE_OUTPUT_DIR);
    strcat(filepath, filename);
    // printf("%s\n", filepath); 
    FILE *fp;
   if ((fp = fopen(filepath,"w")) == NULL){
      return -1;
    }
    for (int j=cursor; j < (cursor + pixels_por_slice); j++){
      fprintf(fp, "%d", imagem[j]);
      if (((j+1) % WIDTH) == 0) {
        fprintf(fp, "\n");
      } else {
        fprintf(fp, ","); 
      }
    }
    cursor += pixels_por_slice;
    fclose(fp);
  }
   return 0;
}

// *********************************************************************
// obtém posição de um voxel no vetor linearizado
// *********************************************************************z
__host__ __device__ 
int getFlat(int x, int y, int z){
  int offset_y = WIDTH;
  int offset_z = WIDTH * WIDTH;  
  int flat = z * offset_z + y * offset_y + x;
  return flat;
}

// *********************************************************************
// obtém coordenadas de um elemento do vetor linearizado
// *********************************************************************
__host__ __device__ 
int getCoord(int flat, int *x, int *y, int *z){
  int offset_y = WIDTH;
  int offset_z = WIDTH * WIDTH;
  *z = flat / (offset_z);
  *y = (flat - ((*z) * (offset_z)))/offset_y;
  *x =  flat - ((*z) * (offset_z)) - ((*y) * offset_y);
  return 0;
}

// *********************************************************************
// calcula o pixel semente
// *********************************************************************
int calculateSeed(int *imagedata, int depth){
  // Inicialmente usando uma semente apenas.
  // Para identificar a semente incial utilizei o seguinte critério:
  // No corte central, busca na linha 255, a partir da coluna 255 o primeiro pixel entre -600 e -700 (tipicamente pulmão)
  int x = WIDTH / 2; //256
  int y = WIDTH / 2; // 256
  int z = depth / 2;
  int pos_seed = -1;
  for (int i = x; i < WIDTH; i++){
    int flat = getFlat(i, y, z);
    //printf("imagedata[%d] (%d, %d, %d): %d\n", flat, i, y, z, imagedata[flat]);
    if (imagedata[flat] > HU_PULMAO_MIN && imagedata[flat] < HU_PULMAO_MAX){
      pos_seed = flat;
      break;
    }
  }
  return (pos_seed);
}


// *********************************************************************
// funcao para verificar se é um pixel vizinho a região
// *********************************************************************
__host__ __device__ 
bool isNeighbor(int index, int *regiondata, int depth){
  int x; int y; int z;
  getCoord(index, &x, &y, &z);
  // printf("calcula feature: %d, %d, %d\n", x, y, z);
  for (int k = z-1; k <= z + 1; k++){
    for (int j = y-1; j <= y + 1; j++){
      for (int i = x-1; i <= x + 1; i++){
          if (((k > 0) && (k < depth)) && ((j > 0) && (j < WIDTH)) && ((i > 0) && (i < WIDTH))){ // testa se está dentro da imagem
            // printf("(k, j, i): (%d, %d, %d)\n", k, j, i);
            int index_neighbor = getFlat(i, j, k);
            if (index_neighbor != index)  // testa se não é o próprio elemento
              if (regiondata[index_neighbor] == 1)  // se um dos vizinhos é 1 ele é um vizinho
                return true;
          }
      }
    }
  }  
  return false;
}

// *********************************************************************
// MIN-MAX HU normalization
// *********************************************************************
__host__ __device__ 
float normalizeHU(int hu){
  if (hu<MIN_HU) 
    hu = MIN_HU;
  else if (hu > MAX_HU){
    hu = MAX_HU;
  } 
  return ((float)abs(hu-MIN_HU))/abs(MAX_HU-MIN_HU);
}

// *********************************************************************
// funcao para calcular o vetor de caracteristicas  (HU, MEAN, MIN, MAX, CVE)
// *********************************************************************
__host__ __device__ 
int calculateFeatures(int index, int *pixeldata, int depth, float *vector){
  vector[0] = normalizeHU(pixeldata[index]); //HU
  vector[1] = 0; // MEAN
  vector[2] = 0; // MIN
  vector[3] = 0; // MAX
  vector[4] = 0; // CVE (to be implemented)
  int x; int y; int z;
  getCoord(index, &x, &y, &z);
  // printf("calcula feature: %d, %d, %d\n", x, y, z);
  float min = 1;
  float max = 0;
  float sum = 0;
  float qtde = 0;
  for (int k = z-1; k <= z + 1; k++){
    for (int j = y-1; j <= y + 1; j++){
      for (int i = x-1; i <= x + 1; i++){
          if (((k > 0) && (k < depth)) && ((j > 0) && (j < WIDTH)) && ((i > 0) && (i < WIDTH))){
            // printf("(k, j, i): (%d, %d, %d)\n", k, j, i);
            float hu = normalizeHU(pixeldata[getFlat(i, j, k)]);
            sum+=hu;
            if (hu < min) min = hu;
            if (hu > max) max = hu;
            qtde++;
          }
      }
    }
  }
  vector[1] = (sum/qtde); // MEAN
  vector[2] = min; // MIN
  vector[3] = max; // MAX  
  return 0;
}

__device__
float calculateDistance(float *vector, float *seed){
  float sum = 0;
  for (int i = 0; i < NUM_FEATURES; i++){
    sum += pow((vector[i] - seed[i]), 2);
  }
  return (float)sqrt(sum);
}

__global__ 
void regionGrowing(int *imagedata, int *regiondata,  float *seed_vector, int *incluidos, int depth){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x < WIDTH) && (y < WIDTH) && (z < depth)){
    int i = getFlat(x, y, z);
    if ((regiondata[i] != 1) && (isNeighbor(i, regiondata, depth))){
      float vector[NUM_FEATURES];
      calculateFeatures(i, imagedata, depth, vector);
      float distance = calculateDistance(vector, seed_vector);
      //printf("[hu, mean, min, max, cve]: [%f, %f, %f, %f, %f] :: distance:=%f\n", vector[0], vector[1], vector[2], vector[3], vector[4], distance );
      if (distance < LIMIAR){
        regiondata[i] = 1;
        *incluidos += 1;
      }
    }
  }
}

__global__ 
void regionMask(int *imagedata, int *regiondata, int depth){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x < WIDTH) && (y < WIDTH) && (z < depth)){
    int i = getFlat(x, y, z);
    if  (regiondata[i] == 0) {
       regiondata[i] = MIN_HU;
    } else {
       regiondata[i] = imagedata[i];
    }   
  }
}

// *********************************************************************
// função principal do programa
// *********************************************************************
int main(void)
{
  
  // 1. inicializa variáveis no host
  int num_slices = 0;
  num_slices = countSlices();
  int num_elementos = num_slices * WIDTH * WIDTH;
  size_t sizect = num_elementos * sizeof(int);
  int *h_imagedata = (int *)malloc(sizect);
  int *h_regiondata = (int *)malloc(sizect);
  // inicializa vetor da regiao com zeros
  for (int i = 0; i < num_elementos; i++) h_regiondata[i] = 0;
  printf(">>> resumo da TC \n");  
  printf("num slices da TC: %d\n", num_slices);  
  printf("tamanho da TC (elementos): %d\n", num_elementos);
  printf("tamanho da TC (bytes): %lu\n", sizect);
  
  // 2. carrega os cortes na memoria principal
  printf(">>> carregando a tomografia na memória principal \n");  
  if (loadCT(h_imagedata) != 0){
    printf("erro ao carregar arquivos da tomografia\n");
    return(-1);
  }

  // 3. aloca as variaveis na memoria do device 
  int *d_imagedata;
  cudaMalloc((void **)&d_imagedata, sizect);
  int *d_regiondata;
  cudaMalloc((void **)&d_regiondata, sizect);  
  
  // 4. identifica o pixel semente e calcula vetor de caracteristicas (HU, MEAN, MIN, MAX, CVE)
  printf(">>> identificando a semente\n");    
  int index_seed = 0;
  if ((index_seed = calculateSeed(h_imagedata, num_slices)) < 0){
    printf("erro ao calcular o pixel semente\n");
    return(-1);
  }
  if (index_seed == 0){
     printf("não obteve a semente para o crescimento de região\n");
    return(-1);
  }else{
     printf("posição da semente: %d\n", index_seed);    
  }
  h_regiondata[index_seed] = 1;
  // calcula vetor de caracteristicas da semente(HU, MEAN, MIN, MAX, CVE)
  size_t size_vector = 5 * sizeof(float);
  float *h_seed_vector = (float *)malloc(size_vector);
  float *d_seed_vector;
  cudaMalloc((void **)&d_seed_vector, size_vector);
  calculateFeatures(index_seed, h_imagedata, num_slices, h_seed_vector);
  
  // 5. copia os dados na memória do device
  cudaMemcpy(d_imagedata, h_imagedata, sizect, cudaMemcpyHostToDevice);
  cudaMemcpy(d_regiondata, h_regiondata, sizect, cudaMemcpyHostToDevice);
  cudaMemcpy(d_seed_vector, h_seed_vector, size_vector, cudaMemcpyHostToDevice);
             
  // 4. inicia loop com o crescimento de regiao e roda ate que novos pixels nao sejam mais incluidos
   int *h_incluidos = (int *)malloc(sizeof(int));
   int *d_incluidos;
   cudaMalloc((void **)&d_incluidos, sizeof(int));
  
   // define o número de blocos e threads
   dim3 dimBlock(16, 16, 4);
   dim3 dimGrid(32, 32, (num_slices+4)/4); 
  int iteracao = 0;
  do{
      *h_incluidos = 0;
      cudaMemcpy(d_incluidos, h_incluidos, sizeof(int), cudaMemcpyHostToDevice);
      regionGrowing<<<dimGrid,dimBlock>>>(d_imagedata, d_regiondata, d_seed_vector, d_incluidos, num_slices);
      cudaDeviceSynchronize();
      cudaMemcpy(h_incluidos, d_incluidos, sizeof(int), cudaMemcpyDeviceToHost);
      printf("%d) incluidos=%d\n", iteracao++, *h_incluidos); //debug
   } while(*h_incluidos != 0);

  // 5. Kernel que aplica uma máscara na imagem original para destacar a área obtida com o crescimento de região
  //    O resultado é armazenado na próxima mascara (d_regiondata)
  regionMask<<<dimGrid,dimBlock>>>(d_imagedata, d_regiondata, num_slices);
  
  // 6. copia resultado para memoria principal
  cudaMemcpy(h_regiondata, d_regiondata, sizect, cudaMemcpyDeviceToHost);
  
  // 7. salva em disco
  printf(">>> carregando a região em disco \n");  
  if (saveCT(h_regiondata, num_slices) != 0){
    printf("erro ao salvar o resultado em disco\n");
    return(-1);
  }
  
  // 8. limpeza
  free (h_imagedata);
  free (h_regiondata);
  free(h_seed_vector);
  free(h_incluidos);
  cudaFree(d_imagedata);
  cudaFree(d_regiondata);  
  cudaFree(d_seed_vector);
  cudaFree(d_incluidos);
  
  printf("Done\n");
  return 0;
}
