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
#include "dem/particle_insertion.h"
#include "dem/particle_wall_contact_detection.h"
#include "dem/particle_wall_contact_force.h"

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


  const unsigned int n_properties = 24;
  float x_min = -0.05; float y_min = -0.05; float z_min = -0.05;
  float x_max = 0.05;  float y_max = 0.05;  float z_max = 0.05;
  double dp = 0.005;
  int nInsert = 10;
  int rhop = 2500;
  Point<dim> g = {0, 0, -9.81};
  float dt = 0.00001;
  int Yp = 50000000; int Yw = 50000000;
  float vp = 0.3; float vw = 0.3;
  float ew = 0.5;
  float muw = 0.5;
  float murw = 0.1;

  Particles::ParticleHandler<dim, dim> particle_handler(tr,
                                                        mapping,
                                                        n_properties);
  std::vector<std::tuple<int,
                         typename Triangulation<dim>::active_cell_iterator,
                         int,
                         Point<dim>,
                         Point<dim>>>
    boundaryCellInfo;


  Point<dim> position1 = {-0.998, 0, 0};
  int      id1       = 0;

  Particles::Particle<dim> particle1(position1, position1, id1);
  typename Triangulation<dim, dim>::active_cell_iterator cell1 =
    GridTools::find_active_cell_around_point(tr, particle1.get_location());
  Particles::ParticleIterator<dim, dim> pit1 =
    particle_handler.insert_particle(particle1, cell1);
  pit1->get_properties()[0]  = id1;
  pit1->get_properties()[1]  = 1;
  pit1->get_properties()[2]  = dp;
  pit1->get_properties()[3]  = rhop;
  pit1->get_properties()[4]  = position1[0];
  pit1->get_properties()[5]  = position1[1];
  pit1->get_properties()[6]  = position1[2];
  pit1->get_properties()[7]  = 0.01;
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


  ParticleWallContactDetection<dim> pw1;
  pw1.boundaryCellsAndFaces(tr, boundaryCellInfo);
  std::vector<std::tuple<std::pair<typename Particles::ParticleIterator<dim, dim>, int>,
                         Point<dim>,
                         Point<dim>>>
    pwContactList;
  pwContactList = pw1.pwcontactlist(boundaryCellInfo, particle_handler);

  std::vector<std::tuple<std::pair<Particles::ParticleIterator<dim, dim>, int>,
                         Point<dim>,
                         Point<dim>,
                         double,
                         double,
                         double,
                         Point<dim>,
                         double>>
    pwContactInfo;
  pw1.pwFineSearch(pwContactList,
                   pwContactInfo,
                   dt);

  ParticleWallContactForce<dim,dim> pwcf1;
  pwcf1.pwNonLinearCF(pwContactInfo, vp, Yp, vw, Yw, muw, murw);


  auto particle = particle_handler.begin();

  deallog << "The contact force acting on particle 1 is: "
          << particle->get_properties()[13] << " N " << std::endl;
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  initlog();
  test<3>();
}
