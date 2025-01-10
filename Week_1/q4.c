//4) WAP in MPI to toggle the character of a given string indexed by the rank of the process

#include <stdio.h>
#include <mpi.h>
#include <string.h>

void toggle(char* word,int rank)
{
  char new[strlen(word) +1];
  strcpy(new,word);
  if(strlen(word)<= rank)
  {
  	printf("P%d: Word too short\n",rank);
  	return;
  }
  
  if(word[rank]>=65 && word[rank]<=90)
   	new[rank]=32+word[rank];
  
  else if(word[rank]>=97 && word[rank]<=122)
  	new[rank]=word[rank]-32;
  
  else
  {
  	printf("P%d: %c invalid char\n",rank,word[rank]);
	return;
  }
  
  printf("P%d: Toggled %s to %s\n",rank,word,new);	
}

int main(int argc,char* argv[])
{
	int rank;
	char word[100];
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	strcpy(word,argv[1]); //to read from command line
	toggle(word,rank);
	MPI_Finalize();
	return 0;
	
}