#include <mpi.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ctime>

#include</usr/lib64/gcc/x86_64-suse-linux/4.5/include/omp.h>


using namespace std;


void countfit(void);
int distance(vector<int> tour);		//prototype
void evolve(void);
vector<int> crosstour(vector<int> a, vector<int> b);
vector<int> mutateTour(vector<int> tour);
vector<int> RandomTour(void);
void MigratePopulation(void);
int algorithm(void);

int comm_size, my_rank;
int city = 0;
int population_size = 1000;
int max_generations = 1000;			//basic setting
int migration_size = 100;
double sum;
int first_index, first_cost;
int second_index, second_cost;
int path[500][500];
vector< vector<int> > population;
vector<int> cost(1000,0);
vector<double> fitness(1000,0);


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    srand(time(NULL));
    double start,end;
    city = atoi(argv[2]);
	FILE *fp = fopen(argv[1], "r");
    if (fp!=NULL) {
        for (int i = 1; i <= city; i++) {
            for (int k = 1; k <= city; k++)
                fscanf(fp, "%d", &path[i][k]);	//get the file and basic cost
        }

        fclose(fp);
    }
    population.resize(1000);
    MPI_Barrier(MPI_COMM_WORLD);
 
    start = omp_get_wtime();
        algorithm();		//algorithm
    end = omp_get_wtime();  //count time
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0){
    	printf("Optimal cost: %d\n",distance(population[first_index]));	//print the result and optimal cost
		printf("the best tour :");
		for (int i = 0; i < city; ++i)
	        printf("%d -> ", population[first_index][i]);
    	printf("%d\n", population[first_index][0]);
        printf("time: %.4f\n", end - start);
    }

    MPI_Finalize();

    return 0;
}

void countfit(void) {
    sum = 0;
    first_index = 0;
    second_index = 0;
    first_cost = 100000000;
    second_cost = 100000000;

    #pragma omp parallel for
    for (int i = 0; i < population_size; i++)
	{
        cost[i] = distance(population[i]);	//compare the cost and set the lower cost
        fitness[i] = 1.0 / cost[i];
        sum += fitness[i];

        if (cost[i] < first_cost) 
		{
            second_index = first_index;
            first_index = i;
            second_cost = first_cost;
            first_cost = cost[i];
        } 
		else if (cost[i] < second_cost) 
		{
            second_index = i;
            second_cost = cost[i];
        }
    }
}


int distance(vector<int> tour) {
    int result = 0;

    for (int i = 1; i < city; i++)
        result += path[tour[i - 1]][tour[i]];
    result += path[tour[city - 1]][tour[0]];		//count the total distance

    return result;
}

vector<int> crosstour(vector<int> a, vector<int> b) {
    vector<int> result(city);

    bool contained[1000];
    int start_index = rand() % city;
    int end_index = rand() % city;
	for(int i=0;i<1000;i++)
	{
		contained[i] = false;		//if ever find the city go to next
	}

    while(start_index == end_index)
        end_index = rand() % city;

    if (start_index > end_index)
        swap(start_index, end_index);

    for (int i = start_index; i <= end_index; i++)
    {
    	result[i] = a[i];
		contained[result[i]] = true;
	} 

    for (int i = 0; i < city; ++i)
	{
        if (!contained[b[i]])
		{
            for (int k = 0; k < city; ++k)
			{
                if (!result[k])
				{
					result[k] = b[i];
                    contained[result[k]] = true;
                    break;
                }
            }
        }
    }

    return result;
}

vector<int> mutateTour(vector<int> tour) {
    vector<int> result = tour;

    int first_index = rand() % city;
    int second_index = rand() % city;

    while(first_index == second_index)
        second_index = rand() % city;

    swap(result[first_index], result[second_index]);

    return result;
}

vector<int> RandomTour(void) {
    int result_index = (rand() % 2) ? first_index : second_index;
    double prob = (double) rand() / RAND_MAX;

    while(prob > fitness[result_index] / sum)
	{
        prob -= fitness[result_index] / sum;	//prob minus fitness if ever go their
        result_index = (result_index + 1) % population_size;
    }

    return population[result_index];
}

void evolve(void) {
    vector< vector<int> > new_population(population_size);
    new_population[0] = population[first_index];
    new_population[1] = population[second_index];

    #pragma omp parallel for
    for (int i = 2; i < population_size; i += 2)
	{
        vector<int> parent1 = RandomTour();
        vector<int> parent2 = RandomTour();
        vector<int> child1 = parent1;
        vector<int> child2 = parent2;

        if ((double)rand() / RAND_MAX < 0.18f)	//use the prob to decide which city to go
		{
            child1 = crosstour(parent1, parent2);
            child2 = crosstour(parent2, parent1);
        }

        if ((double)rand() / RAND_MAX < 0.18f)
            child1 = mutateTour(child1);
        if ((double)rand() / RAND_MAX < 0.18f)
            child2 = mutateTour(child2);

        new_population[i] = child1;
        new_population[i + 1] = child2;
    }

    population = new_population;

    countfit();
}

void MigratePopulation(void) {
    vector< vector<int> > new_population(1000);

    new_population[0] = population[first_index];
    new_population[1] = population[second_index];

    for (int i = 2; i < population_size - migration_size; i++)
        new_population[i] = RandomTour();

    for (int i = population_size - migration_size; i < population_size; i++)
	{
        vector<int> selected = RandomTour();
        vector<int> received(city);

        int prev_rank = (my_rank - 1 + comm_size) % comm_size;	//to the city
        int next_rank = (my_rank + 1) % comm_size;

        if (my_rank % 2)
		{
            MPI_Recv(&received[0], city, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&selected[0], city, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
        } else
		{
            MPI_Send(&selected[0], city, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
            MPI_Recv(&received[0], city, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        new_population[i] = received;
    }

    population = new_population;

    countfit();
}

int algorithm(void) {
    vector<int> tour(city);//initial the population
    for (int i = 1; i <= city; ++i)
        tour[i - 1] = i;

    for (int i = 0; i < population_size; i++)
	{
        population[i] = tour;
    }
    countfit();
    
    for (int i = 0; i < max_generations; i++)	//the number to run the algorithm
	{
        if (comm_size != 1 && i != 0 && i % 100 == 0)
		{
            MPI_Barrier(MPI_COMM_WORLD);
            MigratePopulation();
        }
		evolve();
    }

    return 0;
}
