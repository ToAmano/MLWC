#include<stdio.h>
#include<omp.h>

int main(){
  int a[1000];
  int b[1000];
  int c[1000];
  int i;

  #pragma omp parallel for
  for(i=0;i<1000;i++){
    a[i] = i;
    b[i] = 1;
    c[i] = a[i] + b[i];
  }

  for(i=0;i<1000;i++){
    printf("%d\n",c[i]);
  }

  return 0;
}
