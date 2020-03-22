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
#include "dem/find_cell_neighbors.h"
#include "dem/pp_broad_search.h"
#include "dem/pp_contact_info_struct.h"
#include "dem/pp_fine_search.h"

using namespace dealii;

template <int dim>
void
test()
{
  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, -1, 1, true);
  int numRef = 2;
  triangulation.refine_global(numRef);
  int           cellNum = triangulation.n_active_cells();
  MappingQ<dim> mapping(1);

  const unsigned int n_properties = 21;

  Particles::ParticleHandler<dim> particle_handler(triangulation,
                                                   mapping,
                                                   n_properties);

  std::vector<std::set<typename Triangulation<dim>::active_cell_iterator>>
    cellNeighbor;

  FindCellNeighbors<dim> cn1;
  cellNeighbor = cn1.find_cell_neighbors(triangulation);

  PPBroadSearch<dim> ppbs;
  PPFineSearch<dim>  ppfs;

  int      num_Particles = 2;
  Point<3> position1     = {0.4, 0, 0};
  int      id1           = 0;
  Point<3> position2     = {0.40499, 0, 0};
  int      id2           = 1;
  float    dt            = 0.00001;

  Particles::Particle<dim> particle1(position1, position1, id1);
  typename Triangulation<dim>::active_cell_iterator cell1 =
    GridTools::find_active_cell_around_point(triangulation,
                                             particle1.get_location());
  Particles::ParticleIterator<dim> pit1 =
    particle_handler.insert_particle(particle1, cell1);
  pit1->get_properties()[0]  = id1;
  pit1->get_properties()[1]  = 1;
  pit1->get_properties()[2]  = 0.005;
  pit1->get_properties()[3]  = 2500;
  pit1->get_properties()[4]  = 0;
  pit1->get_properties()[5]  = 0;
  pit1->get_properties()[6]  = 0;
  pit1->get_properties()[7]  = 0;
  pit1->get_properties()[8]  = 0;
  pit1->get_properties()[9]  = 0;
  pit1->get_properties()[10] = 0;
  pit1->get_properties()[11] = 0;
  pit1->get_properties()[12] = 0;
  pit1->get_properties()[13] = 0;
  pit1->get_properties()[14] = 0;
  pit1->get_properties()[15] = 0;
  pit1->get_properties()[16] = 1;
  pit1->get_properties()[17] = 1;

  Particles::Particle<dim> particle2(position2, position2, id2);
  typename Triangulation<dim>::active_cell_iterator cell2 =
    GridTools::find_active_cell_around_point(triangulation,
                                             particle2.get_location());
  Particles::ParticleIterator<dim> pit2 =
    particle_handler.insert_particle(particle2, cell2);
  pit2->get_properties()[0]  = id2;
  pit2->get_properties()[1]  = 1;
  pit2->get_properties()[2]  = 0.005;
  pit2->get_properties()[3]  = 2500;
  pit2->get_properties()[4]  = 0;
  pit2->get_properties()[5]  = 0;
  pit2->get_properties()[6]  = 0;
  pit2->get_properties()[7]  = 0;
  pit2->get_properties()[8]  = 0;
  pit2->get_properties()[9]  = 0;
  pit2->get_properties()[10] = 0;
  pit2->get_properties()[11] = 0;
  pit2->get_properties()[12] = 0;
  pit2->get_properties()[13] = 0;
  pit2->get_properties()[14] = 0;
  pit2->get_properties()[15] = 0;
  pit2->get_properties()[16] = 1;
  pit2->get_properties()[17] = 1;

  std::vector<std::pair<Particles::ParticleIterator<dim>,
                        Particles::ParticleIterator<dim>>>
    pairs;
  ppbs.find_PP_Contact_Pairs(particle_handler, cellNeighbor, pairs );

  std::vector<std::map<int, Particles::ParticleIterator<dim>>> inContactPairs(
    num_Particles);
  std::vector<std::map<int, pp_contact_info_struct<dim>>> inContactInfo(
    num_Particles);

  ppfs.pp_Fine_Search(pairs, inContactInfo, dt);

  typename std::map<int, pp_contact_info_struct<dim>>::iterator info_it;
  for (unsigned int i = 0; i < inContactInfo.size(); i++)
    {
      info_it = inContactInfo[i].begin();
      while (info_it != inContactInfo[i].end())
        {
          deallog << "The normal overlap of contacting paritlce is: "
                  << info_it->second.normal_overlap << std::endl;
          deallog << "The normal vector of collision is: "
                  << info_it->second.normal_vector[0] << " "
                  << info_it->second.normal_vector[1] << " "
                  << info_it->second.normal_vector[2] << std::endl;
          deallog << "Normal relative velocity at contact point is: "
                  << info_it->second.normal_relative_velocity << std::endl;
          deallog << "The tangential overlap of contacting paritlce is: "
                  << info_it->second.tangential_overlap << std::endl;
          deallog << "The tangential vector of collision is: "
                  << info_it->second.tangential_vector[0] << " "
                  << info_it->second.tangential_vector[1] << " "
                  << info_it->second.tangential_vector[2] << std::endl;
          deallog << "Tangential relative velocity at contact point is: "
                  << info_it->second.tangential_relative_velocity << std::endl;
          info_it++;
        }
    }
}

int
main(int argc, char **argv)
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  test<3>();
}
