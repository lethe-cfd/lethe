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
#include "dem/contact_info_struct.h"
#include "dem/contact_search.h"
#include "dem/particle_insertion.h"

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


  std::string            filename = "../dem.prm";
  ParameterHandler       prm;
  DEMSolverParameters<3> DEMparam;
  DEMparam.declare(prm);
  prm.parse_input(filename);
  DEMparam.parse(prm);

  const unsigned int n_properties = DEMparam.outputProperties.numProperties;

  Particles::ParticleHandler<dim, dim> particle_handler(tr,
                                                        mapping,
                                                        n_properties);


  std::pair<std::vector<
              std::set<typename Triangulation<dim, dim>::active_cell_iterator>>,
            std::vector<typename Triangulation<dim, dim>::active_cell_iterator>>
    cellNeighbor;

  ContactSearch<dim, dim> cs1;
  cellNeighbor = cs1.findCellNeighbors(tr);

  int      num_Particles = 2;
  Point<3> position1     = {0.4, 0, 0};
  int      id1           = 1;
  Point<3> position2     = {0.40499, 0, 0};
  int      id2           = 2;

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


  Particles::Particle<dim> particle2(position2, position2, id2);
  typename Triangulation<dim, dim>::active_cell_iterator cell2 =
    GridTools::find_active_cell_around_point(tr, particle2.get_location());
  Particles::ParticleIterator<dim, dim> pit2 =
    particle_handler.insert_particle(particle2, cell2);
  pit2->get_properties()[0]  = id2;
  pit2->get_properties()[1]  = 1;
  pit2->get_properties()[2]  = DEMparam.physicalProperties.diameter;
  pit2->get_properties()[3]  = DEMparam.physicalProperties.density;
  pit2->get_properties()[4]  = position2[0];
  pit2->get_properties()[5]  = position2[1];
  pit2->get_properties()[6]  = position2[2];
  pit2->get_properties()[7]  = 0;
  pit2->get_properties()[8]  = 0;
  pit2->get_properties()[9]  = 0;
  pit2->get_properties()[10] = 0;
  pit2->get_properties()[11] = 0;
  pit2->get_properties()[12] = 0;
  pit2->get_properties()[13] = 0;
  pit2->get_properties()[14] = 0;
  pit2->get_properties()[15] = 0;
  pit2->get_properties()[16] = 0;
  pit2->get_properties()[17] = 0;
  pit2->get_properties()[18] = 0;
  pit2->get_properties()[19] = 1;
  pit2->get_properties()[20] = 1;

  std::vector<std::pair<Particles::ParticleIterator<dim, dim>,
                        Particles::ParticleIterator<dim, dim>>>
    pairs;
  pairs = cs1.findContactPairs(particle_handler, tr, cellNeighbor.first);


  std::vector<std::map<int, Particles::ParticleIterator<dim, dim>>>
                                                          inContactPairs(num_Particles);
  std::vector<std::map<int, ContactInfoStruct<dim, dim>>> inContactInfo(
    num_Particles);


  cs1.fineSearch(pairs,
                 inContactPairs,
                 inContactInfo,
                 DEMparam.simulationControl.dt,
                 particle_handler);

  typename std::map<int, ContactInfoStruct<dim, dim>>::iterator info_it;
  for (unsigned int i = 0; i < inContactInfo.size(); i++)
    {
      info_it = inContactInfo[i].begin();
      while (info_it != inContactInfo[i].end())
        {
          deallog << "The normal overlap of contacting paritlce is: "
                  << info_it->second.normOverlap << std::endl;
          deallog << "The normal vector of collision is: "
                  << info_it->second.normVec[0] << " "
                  << info_it->second.normVec[1] << " "
                  << info_it->second.normVec[2] << std::endl;
          deallog << "Normal relative velocity at contact point is: "
                  << info_it->second.normRelVel << std::endl;
          deallog << "The tangential overlap of contacting paritlce is: "
                  << info_it->second.tangOverlap << std::endl;
          deallog << "The tangential vector of collision is: "
                  << info_it->second.tangVec[0] << " "
                  << info_it->second.tangVec[1] << " "
                  << info_it->second.tangVec[2] << std::endl;
          deallog << "Tangential relative velocity at contact point is: "
                  << info_it->second.tangRelVel << std::endl;
          info_it++;
        }
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  initlog();
  test<3>();
}
