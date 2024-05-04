#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>

#define frand() (rand()/(RAND_MAX+1.0))

#define NUMDIMS 3
#define NUMATOMS 1000
#define NUMSTEPS 16
#define MASS 1.0
#define DT 1.0e-4
#define CUTOFF 0.5

#define PI2 (3.1415926/2.0)
#define MIN(x,y) ((x)>(y)?(y):(x))

#define V(x) (sin(MIN(x, PI2))*sin(MIN(x, PI2)))
#define DV(x) (2.0*sin(MIN(x, PI2))*cos(MIN(x, PI2)))

typedef struct AtomInfo{
    double mass;
    double position[NUMDIMS];
    double velocity[NUMDIMS];
    double acceleration[NUMDIMS];
    double force[NUMDIMS];
}AtomInfo;

void initialize(AtomInfo *atoms);
void compute(AtomInfo *atoms, double *pPotential, double *pKinetic);
void update(AtomInfo *atoms);

int main(int argc, char *argv[]){
    AtomInfo atoms[NUMATOMS];
    double potential, kinetic, E0;
    int i;

    // Set initial positions, velocities, and accelerations
    initialize(atoms);

    // Compute the forces and energies
    compute(atoms, &potential, &kinetic);

    E0 = potential + kinetic;

    printf("Potential,\tKinetic,\tChange Rate,\tTimePerStep(ms)\n");
    
    // Main time stepping loop
    for(i=0; i<NUMSTEPS; i++){
        double execTime = -(clock()/(double)CLOCKS_PER_SEC*1000);
        compute(atoms, &potential, &kinetic);
        execTime += (clock()/(double)CLOCKS_PER_SEC*1000);
        printf("%lf, %lf, %lf, %lf\n", potential, kinetic, (potential+kinetic-E0)/E0, execTime);
        update(atoms); 
    }
    
    return EXIT_SUCCESS;
}

// Initialize atoms with random positions and zero velocities/accelerations
void initialize(AtomInfo *atoms){
    int i,j;
    
    for(i=0; i<NUMATOMS; i++){
        AtomInfo *singleOne = atoms + i;
        singleOne->mass = MASS;
        for(j=0; j<NUMDIMS; j++){
            (singleOne->position)[j] = frand()*10;
            (singleOne->velocity)[j] = 0.0;
            (singleOne->acceleration)[j] = 0.0;
        }
    }
}

// Compute forces and energies
void compute(AtomInfo *atoms, double *pPotential, double *pKinetic){
    int i, j, k;
    
    double potential = 0.0;
    double kinetic = 0.0;
    
    #pragma acc parallel loop private(i, j, k) reduction(+:potential, kinetic)
    for(i=0; i<NUMATOMS; i++){
        AtomInfo *atomA = atoms+i;
        for(k=0; k<NUMDIMS; k++)
            (atomA->force)[k] = 0.0;

        #pragma acc loop vector independent // Parallelize the outer loop
        for(j=i+1; j<NUMATOMS; j++){ // Iterate from i+1 to avoid redundant computations
            AtomInfo *atomB = atoms+j;
            // Compute distance between atomA and atomB
            double d = distance(atomA, atomB);
            if(d > CUTOFF){
                double force = DV(d) / d;
                potential += 0.5 * V(d);
                // Update forces for atomA and atomB
                for(k=0; k<NUMDIMS; k++){
                    double delta = (atomA->position)[k] - (atomB->position)[k];
                    (atomA->force)[k] -= delta * force;
                    (atomB->force)[k] += delta * force;
                }
            }
        }
        kinetic += calKinetic(atomA);
    }
    *pPotential = potential;
    *pKinetic = kinetic;
}

// Compute the Euclidean distance between two atoms
double distance(AtomInfo *atomA, AtomInfo *atomB){
    double dist = 0.0;
    int i;
    for(i=0; i<NUMDIMS; i++){
        double abDist = (atomA->position)[i] - (atomB->position)[i];
        dist += abDist*abDist;
    }
    return sqrt(dist);
}

// Calculate kinetic energy of an atom
double calKinetic(AtomInfo *atom){
    double kinetic = 0.0;
    int i;
    for(i=0; i<NUMDIMS; i++){
        kinetic += (atom->velocity)[i] * (atom->velocity)[i];
    }
    return kinetic*0.5*(atom->mass);
}

// Update positions and velocities of atoms
void update(AtomInfo *atoms){
    int i, j;
    
    #pragma acc parallel loop private(i, j)
    for(i=0; i<NUMATOMS; i++){
        AtomInfo *one = atoms+i;
        for(j=0; j<NUMDIMS; j++){
            (one->position)[j] += (one->velocity)[j]*DT + 0.5*DT*DT*(one->acceleration)[j];
            (one->velocity)[j] += 0.5*DT*((one->force)[j]/one->mass + (one->acceleration)[j]); 
        }
    }
}

