

#include <deal.II/base/array_view.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/partitioner.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>

#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#include "dem/contact_search.h"
#include "dem/dem_iterator.h"
#include "dem/parameters_dem.h"
#include "dem/particle_insertion.h"
#include "dem/particle_wall_contact_detection.h"



using namespace dealii;


template <int dim>
void
initilization()
{
  std::string filename;
  filename = "dem.prm";
  ParameterHandler prm;
  ParametersDEM<dim> DEMparam;
  DEMparam.declare(prm);
  prm.parse_input(filename);
  DEMparam.parse(prm);

  std::vector<std::tuple<std::string, int>> properties(
    DEMparam.outputProperties.numProperties);
  properties[0]  = std::make_tuple("id", 1);
  properties[1]  = std::make_tuple("type", 1);
  properties[2]  = std::make_tuple("Diam", 1);
  properties[3]  = std::make_tuple("Dens", 1);
  properties[4]  = std::make_tuple("Pos", dim);
  properties[5]  = std::make_tuple("Pos", 1);
  properties[6]  = std::make_tuple("Pos", 1);
  properties[7]  = std::make_tuple("Vel", dim);
  properties[8]  = std::make_tuple("Vel", 1);
  properties[9]  = std::make_tuple("Vel", 1);
  properties[10] = std::make_tuple("a", dim);
  properties[11] = std::make_tuple("a", 1);
  properties[12] = std::make_tuple("a", 1);
  properties[13] = std::make_tuple("f", dim);
  properties[14] = std::make_tuple("f", 1);
  properties[15] = std::make_tuple("f", 1);
  properties[16] = std::make_tuple("w", dim);
  properties[17] = std::make_tuple("w", 1);
  properties[18] = std::make_tuple("w", 1);
  properties[19] = std::make_tuple("mass", 1);
  properties[20] = std::make_tuple("MOI", 1);


  // total number of particles in the system
  int nPart = 0;
  // DEM clock
  int   DEM_step = 0;
  float DEM_time = 0;

  parallel::distributed::Triangulation<dim, dim> tr(MPI_COMM_WORLD);

  Particles::Particle<dim> particle;
  GridGenerator::hyper_cube(tr, -0.1, 0.1, true);
  int numRef = 3;
  tr.refine_global(numRef);
  auto &mappinggg = StaticMappingQ1<3, 3>::mapping;

  Particles::ParticleHandler<3, 3> particle_handler(
    tr, mappinggg, DEMparam.outputProperties.numProperties);
  Particles::PropertyPool propPool(DEMparam.outputProperties.numProperties);


  int cellNum = tr.n_active_cells();


  std::pair<std::vector<std::set<Triangulation<3>::active_cell_iterator>>,
            std::vector<Triangulation<3>::active_cell_iterator>>
                cellNeighbor;
  ContactSearch cs1;
  cellNeighbor = cs1.findCellNeighbors(cellNum, tr);


  DEM_iterator iter1;
  std::vector<std::tuple<std::pair<Particles::ParticleIterator<3, 3>,
                                   Particles::ParticleIterator<3, 3>>,
                         double,
                         Point<3>,
                         double,
                         Point<3>,
                         double,
                         double>>
    contactInfo;
  std::vector<std::tuple<std::pair<Particles::ParticleIterator<3, 3>, int>,
                         Point<3>,
                         Point<3>,
                         double,
                         double,
                         double,
                         Point<3>,
                         double>>
    pwContactInfo;


  std::vector<std::tuple<int,
                         Triangulation<3>::active_cell_iterator,
                         int,
                         Point<3>,
                         Point<3>>>
                               boundaryCellInfo;
  ParticleWallContactDetection pw1;
  pw1.boundaryCellsAndFaces(tr, boundaryCellInfo);

  // dem engine iterator:
  while (DEM_step < DEMparam.simulationControl.tFinal)
    {
      iter1.engine(nPart,
                   particle_handler,
                   tr,
                   DEM_step,
                   DEM_time,
                   DEMparam,
                   cellNeighbor,
                   contactInfo,
                   boundaryCellInfo,
                   pwContactInfo,
                   properties,
                   propPool);
    }
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, numbers::invalid_unsigned_int);

  initilization<3>();

  return 0;
}
