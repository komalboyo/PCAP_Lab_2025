// Write a program in cude to count the number of times a given word 
// is repeated in a sentence. Use atomic functions

#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

#define MAX_WORDS 1024  
#define WORD_LENGTH 32  


__device__ bool strcmp_cuda(const char *a, const char *b) {
    while (*a && (*a == *b)) {
        a++;
        b++;
    }
    return (*a == '\0' && *b == '\0');  

}


__global__ void count_word_kernel(char *words, int num_words, char *target, int *count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_words) return;

    char *word = words + idx * WORD_LENGTH;  
    if (strcmp_cuda(word, target)) {
        atomicAdd(count, 1);  
    }
}

int main() {
    const char *sentence = "hi my name is komal. hi hi hi";
    const char *target_word = "hi";

    
    char h_words[MAX_WORDS * WORD_LENGTH] = {0}; 
    int num_words = 0;

   char temp_sentence[256];
    strcpy(temp_sentence, sentence);
    char *token = strtok(temp_sentence, " ");

    while (token != NULL && num_words < MAX_WORDS) {
        strncpy(&h_words[num_words * WORD_LENGTH], token, WORD_LENGTH - 1);
        h_words[num_words * WORD_LENGTH + WORD_LENGTH - 1] = '\0'; // Null terminate
        num_words++;
        token = strtok(NULL, " ");
    }

    char *d_words, *d_target;
    int *d_count, h_count = 0;

    cudaMalloc((void**)&d_words, MAX_WORDS * WORD_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_target, WORD_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_words, h_words, MAX_WORDS * WORD_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_word, WORD_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);

    
    int threads_per_block = 256;
    int blocks = (num_words + threads_per_block - 1) / threads_per_block;
    count_word_kernel<<<blocks, threads_per_block>>>(d_words, num_words, d_target, d_count);

    
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Word '%s' appeared %d times in the sentence.\n", target_word, h_count);

    cudaFree(d_words);
    cudaFree(d_target);
    cudaFree(d_count);

    return 0;
}
