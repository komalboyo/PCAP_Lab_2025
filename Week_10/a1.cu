#include <stdio.h>
#include <cuda.h>

#define MAX_ITEMS 10
#define MAX_PURCHASES 100

__global__ void calculatePurchase(int *itemPrices, int *purchases, int *total, int totalPurchases) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < totalPurchases) {
        atomicAdd(total, itemPrices[purchases[idx]]);
    }
}

int main() {
    int itemPrices[MAX_ITEMS] = {100, 200, 300, 150, 250, 350, 400, 180, 220, 280};
    int *d_itemPrices, *d_purchases, *d_total;

    int N;
    printf("Enter number of friends: ");
    scanf("%d", &N);

    int purchases[MAX_PURCHASES];
    int totalPurchases = 0;

    for (int i = 0; i < N; ++i) {
        int itemsCount;
        printf("Enter number of items purchased by friend %d: ", i + 1);
        scanf(" %d", &itemsCount);
        printf("Enter item indices (0 to 9): ");
        for (int j = 0; j < itemsCount; ++j) {
            int idx;
            scanf("%d", &idx);
            purchases[totalPurchases++] = idx;
        }
    }

    cudaMalloc(&d_itemPrices, MAX_ITEMS * sizeof(int));
    cudaMalloc(&d_purchases, totalPurchases * sizeof(int));
    cudaMalloc(&d_total, sizeof(int));

    cudaMemcpy(d_itemPrices, itemPrices, MAX_ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_purchases, purchases, totalPurchases * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_total, 0, sizeof(int));

    int threads = 256;
    int blocks = (totalPurchases + threads - 1) / threads;

    calculatePurchase<<<blocks, threads>>>(d_itemPrices, d_purchases, d_total, totalPurchases);

    int h_total;
    cudaMemcpy(&h_total, d_total, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Total Purchase Amount by All Friends: ₹%d\n", h_total);

    cudaFree(d_itemPrices);
    cudaFree(d_purchases);
    cudaFree(d_total);

    return 0;
}
