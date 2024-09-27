CC ?= cc
CC_FLAGS := -Wall -fopenmp
SRC_FILES := load.h load.c
#SRC_FILES += main_np.c model_np.c
#SRC_FILES += main_np.c model.c # OpenMP
#SRC_FILES += main.c model_np.c # OpenMPI
SRC_FILES += main.c model.c # OpenMP + OpenMPI

ifdef OMP_NUM_THREADS
DEFINES := -DOMP_NUM_THREADS=$(OMP_NUM_THREADS)
endif
N_MPI_TASKS ?= 1

a.out:  $(SRC_FILES)
	$(CC) $(CC_FLAGS) $(DEFINES) $^

run: clean a.out
	export OMP_NUM_THREADS=$(OMP_NUM_THREADS)
	srun -n $(N_MPI_TASKS) a.out
	@echo "TO DO"

clean:
	-rm a.out load.h.gch
