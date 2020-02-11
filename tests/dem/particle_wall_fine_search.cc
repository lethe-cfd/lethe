#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>

#include <iostream>
#include <vector>

#include "../tests.h"
#include "dem/dem_solver_parameters.h"
#include "dem/particle_wall_contact_detection.h"
#include "dem/find_boundary_cells_information.h"

using namespace dealii;

template <int dim>
void
test()
{
  parallel::distributed::Triangulation<dim, dim> tr(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tr, -1, 1, true);
  int numRef = 2;
  tr.refine_global(numRef);
  int                cellNum = tr.n_active_cells();
  MappingQ<dim, dim> mapping(1);


  std::string              filename = "../dem.prm";
  ParameterHandler         prm;
  DEMSolverParameters<dim> DEMparam;
  DEMparam.declare(prm);
  prm.parse_input(filename);
  DEMparam.parse(prm);

  const unsigned int n_properties = DEMparam.outputProperties.numProperties;

  Particles::ParticleHandler<dim, dim> particle_handler(tr,
                                                        mapping,
                                                        n_properties);


  Point<dim> position1 = {-0.998, 0, 0};
  int        id1       = 0;

  Particles::Particle<dim> particle1(position1, position1, id1);
  typename Triangulation<dim, dim>::active_cell_iterator cell1 =
    GridTools::find_active_cell_around_point(tr, particle1.get_location());
  Particles::ParticleIterator<dim, dim> pit1 =
    particle_handler.insert_particle(particle1, cell1);
  pit1->get_properties()[0]  = id1;
  pit1->get_properties()[1]  = 1;
  pit1->get_properties()[2]  = DEMparam.physicalProperties.diameter;
  pit1->get_properties()[3]  = DEMparam.physicalProperties.density;
  pit1->get_properties()[4]  = position1[0];
  pit1->get_properties()[5]  = position1[1];
  pit1->get_properties()[6]  = position1[2];
  pit1->get_properties()[7]  = 0;
  pit1->get_properties()[8]  = 0;
  pit1->get_properties()[9]  = 0;
  pit1->get_properties()[10] = 0;
  pit1->get_properties()[11] = 0;
  pit1->get_properties()[12] = 0;
  pit1->get_properties()[13] = 0;
  pit1->get_properties()[14] = 0;
  pit1->get_properties()[15] = 0;
  pit1->get_properties()[16] = 0;
  pit1->get_properties()[17] = 0;
  pit1->get_properties()[18] = 0;
  pit1->get_properties()[19] = 1;
  pit1->get_properties()[20] = 1;

  std::vector<boundary_cells_info_struct<dim>> boundaryCellInfo;
  FindBoundaryCellsInformation<dim, dim> boundary_cells_object;
   boundaryCellInfo = boundary_cells_object.find_boundary_cells_information(tr);


  ParticleWallContactDetection<dim> pw1;
  std::vector<
    std::tuple<std::pair<typename Particles::ParticleIterator<dim, dim>, int>,
               Point<dim>,
               Point<dim>>>
    pwContactList;
  pwContactList = pw1.pwcontactlist(boundaryCellInfo, particle_handler);

  std::vector<
    std::tuple<std::pair<typename Particles::ParticleIterator<dim, dim>, int>,
               Point<dim>,
               Point<dim>,
               double,
               double,
               double,
               Point<dim>,
               double>>
    pwContactInfo;
  pw1.pwFineSearch(pwContactList, pwContactInfo, DEMparam.simulationControl.dt);


  for (unsigned int i = 0; i != pwContactInfo.size(); i++)
    {
      deallog << "The normal overlap of contacting paritlce-wall is: "
              << std::get<3>(pwContactInfo[i]) << std::endl;
      deallog << "The normal vector of collision is: "
              << std::get<1>(pwContactInfo[i])[0] << " "
              << std::get<1>(pwContactInfo[i])[1] << " "
              << std::get<1>(pwContactInfo[i])[2] << std::endl;
      deallog << "Normal relative velocity at contact point is: "
              << std::get<4>(pwContactInfo[i]) << std::endl;
      deallog << "The tangential overlap of contacting paritlce-wall is: "
              << std::get<5>(pwContactInfo[i]) << std::endl;
      deallog << "The tangential vector of collision is: "
              << std::get<6>(pwContactInfo[i])[0] << " "
              << std::get<6>(pwContactInfo[i])[1] << " "
              << std::get<6>(pwContactInfo[i])[2] << std::endl;
      deallog << "Tangential relative velocity at contact point is: "
              << std::get<7>(pwContactInfo[i]) << std::endl;
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  initlog();
  test<3>();
}
