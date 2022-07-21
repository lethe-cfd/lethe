
#include <dem/particle_wall_broad_search.h>

using namespace dealii;

template <int dim>
ParticleWallBroadSearch<dim>::ParticleWallBroadSearch()
{}

template <int dim>
void
ParticleWallBroadSearch<dim>::find_particle_wall_contact_pairs(
  const std::map<int, boundary_cells_info_struct<dim>>
    &                                    boundary_cells_information,
  const Particles::ParticleHandler<dim> &particle_handler,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<unsigned int,
                       std::tuple<Particles::ParticleIterator<dim>,
                                  Tensor<1, dim>,
                                  Point<dim>,
                                  types::boundary_id,
                                  unsigned int>>>
    &particle_wall_contact_candidates)
{
  // Clearing particle_wall_contact_candidates (output of this function)
  particle_wall_contact_candidates.clear();

  // Iterating over the boundary_cells_information, which is the output of
  // the find_boundary_cells_information find_boundary_cells_information class.
  // This map contains all the required information of the system boundary
  // cells and faces. In this loop we find the particles located in each of
  // these boundary cells
  for (auto boundary_cells_information_iterator =
         boundary_cells_information.begin();
       boundary_cells_information_iterator != boundary_cells_information.end();
       ++boundary_cells_information_iterator)
    {
      auto boundary_cells_content = boundary_cells_information_iterator->second;
      auto cell                   = boundary_cells_content.cell;

      // Finding particles located in the corresponding cell
      // (boundary_cells_content.cell)
      typename Particles::ParticleHandler<dim>::particle_iterator_range
        particles_in_cell = particle_handler.particles_in_cell(cell);

      const bool particles_exist_in_main_cell = !particles_in_cell.empty();

      // If the main cell is not empty
      if (particles_exist_in_main_cell)
        {
          for (typename Particles::ParticleHandler<dim>::
                 particle_iterator_range::iterator particles_in_cell_iterator =
                   particles_in_cell.begin();
               particles_in_cell_iterator != particles_in_cell.end();
               ++particles_in_cell_iterator)
            {
              // Making the tuple and adding it to the
              // particle_wall_contact_candidates vector. This vector is the
              // output of this function

              std::tuple map_content =
                std::make_tuple(particles_in_cell_iterator,
                                boundary_cells_content.normal_vector,
                                boundary_cells_content.point_on_face,
                                boundary_cells_content.boundary_id,
                                boundary_cells_content.global_face_id);

              particle_wall_contact_candidates[particles_in_cell_iterator
                                                 ->get_id()]
                .insert({boundary_cells_content.global_face_id, map_content});
            }
        }
    }
}

template <int dim>
void
ParticleWallBroadSearch<dim>::find_particle_floating_wall_contact_pairs(
  const std::unordered_map<
    types::particle_index,
    std::set<typename Triangulation<dim>::active_cell_iterator>>
    &                                    boundary_cells_for_floating_walls,
  const Particles::ParticleHandler<dim> &particle_handler,
  const Parameters::Lagrangian::FloatingWalls<dim> &floating_wall_properties,
  const double &                                    simulation_time,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<unsigned int, Particles::ParticleIterator<dim>>>
    &pfw_contact_candidates)
{
  // Clearing pfw_contact_candidates(output of this function)
  pfw_contact_candidates.clear();

  // Iterating over the boundary_cells_for_floating_walls, which is the output
  // of the find_boundary_cells_for_floating_walls function in
  // find_boundary_cells_information class. This unordered_map contains all the
  // boundary cells of floating walls. In this loop
  // we find the particles located in boundary cells of floating
  // walls
  for (auto fw_boundary_cells_information_iterator =
         boundary_cells_for_floating_walls.begin();
       fw_boundary_cells_information_iterator !=
       boundary_cells_for_floating_walls.end();
       ++fw_boundary_cells_information_iterator)
    {
      unsigned int floating_wall_id =
        fw_boundary_cells_information_iterator->first;

      // Checking simulation time for temporary floating walls
      if (simulation_time >=
            floating_wall_properties.time_start[floating_wall_id] &&
          simulation_time <=
            floating_wall_properties.time_end[floating_wall_id])
        {
          auto boundary_cells_content =
            fw_boundary_cells_information_iterator->second;

          for (auto cell = boundary_cells_content.begin();
               cell != boundary_cells_content.end();
               ++cell)
            {
              // Finding particles located in the corresponding cell
              typename Particles::ParticleHandler<dim>::particle_iterator_range
                particles_in_cell = particle_handler.particles_in_cell(*cell);

              const bool particles_exist_in_main_cell =
                !particles_in_cell.empty();

              // If the main cell is not empty
              if (particles_exist_in_main_cell)
                {
                  for (typename Particles::ParticleHandler<
                         dim>::particle_iterator_range::iterator
                         particles_in_cell_iterator = particles_in_cell.begin();
                       particles_in_cell_iterator != particles_in_cell.end();
                       ++particles_in_cell_iterator)
                    {
                      pfw_contact_candidates[particles_in_cell_iterator
                                               ->get_id()]
                        .insert({floating_wall_id, particles_in_cell_iterator});
                    }
                }
            }
        }
    }
}

template <int dim>
void
ParticleWallBroadSearch<dim>::particle_moving_mesh_contact_search(
  const std::vector<std::vector<
    std::pair<typename Triangulation<dim>::active_cell_iterator,
              typename Triangulation<dim - 1, dim>::active_cell_iterator>>>
    &                                    moving_mesh_information,
  const Particles::ParticleHandler<dim> &particle_handler,
  std::map<
    typename Triangulation<dim - 1, dim>::active_cell_iterator,
    std::unordered_map<types::particle_index, Particles::ParticleIterator<dim>>,
    dem_data_containers::cut_cell_comparison<dim>>
    &particle_moving_mesh_contact_candidates,
  std::unordered_map<
    types::global_cell_index,
    std::vector<typename Triangulation<dim>::active_cell_iterator>>
    &cells_total_neighbor_list)
{
  // Clear the candidate container
  particle_moving_mesh_contact_candidates.clear();

  std::vector<Particles::ParticleIterator<dim>> contact_candidates;

  // I am assuming that triangles in different solids have different unique
  // global ids. If it's not the case, we have to modify the code

  // Loop through solids
  for (auto &solid_iterator : moving_mesh_information)
    {
      // Loop through the pairs (first -> background cell, second -> moving
      // cell)
      for (auto moving_mesh_iterator = solid_iterator.begin();
           moving_mesh_iterator != solid_iterator.end();
           ++moving_mesh_iterator)
        {
          // Get background cell
          auto background_cell = moving_mesh_iterator->first;

          // Get cut cells (moving mesh cells)
          auto cut_cells = moving_mesh_iterator->second;

          // Get neighbors of the background cell
          auto cell_list = cells_total_neighbor_list.at(
            background_cell->global_active_cell_index());

          contact_candidates.clear();

          // Loop through neighbors
          for (auto &cell_iterator : cell_list)
            {
              // Find particles located in cell
              typename Particles::ParticleHandler<dim>::particle_iterator_range
                particles_in_cell =
                  particle_handler.particles_in_cell(cell_iterator);

              const bool particles_exist_in_cell = !particles_in_cell.empty();

              // If the main cell is not empty
              if (particles_exist_in_cell)
                {
                  //   std::vector<Point<dim>> triangle;

                  //     for (unsigned int vertex = 0; vertex <
                  //     vertices_per_triangle;
                  //       ++vertex)
                  //    {
                  // Find vertex-floating wall distance
                  //         triangle.push_back(cut_cells->vertex(vertex));
                  //     }

                  // Call calculate_particle_triangle_distance to get the
                  // distance and projection of particles on the triangle
                  // (moving mesh cell)
                  //     auto particle_triangle_information =
                  //        LetheGridTools::calculate_particle_triangle_distance(
                  //          triangle, particles_in_cell, n_particles_in_cell);

                  //       const std::vector<bool> pass_distance_check =
                  //          std::get<0>(particle_triangle_information);
                  //         const std::vector<Point<dim>> projection_points =
                  //           std::get<1>(particle_triangle_information);

                  //         unsigned int particle_counter = 0;

                  // Loop through particles in the main cell and build contact
                  // candidate pairs
                  for (typename Particles::ParticleHandler<
                         dim>::particle_iterator_range::iterator
                         particles_in_cell_iterator = particles_in_cell.begin();
                       particles_in_cell_iterator != particles_in_cell.end();
                       ++particles_in_cell_iterator)
                    {
                      //  if (pass_distance_check[particle_counter])
                      //   {
                      //     auto particle_mesh_info = std::make_tuple(
                      //        particles_in_cell_iterator,
                      //        std::get<2>(particle_triangle_information),
                      //         projection_points[particle_counter]);


                      //  contact_candidates.push_back(particles_in_cell_iterator);

                      particle_moving_mesh_contact_candidates[cut_cells].insert(
                        {particles_in_cell_iterator->get_id(),
                         particles_in_cell_iterator});
                      //  }
                    }
                }
            }

          // particle_moving_mesh_contact_candidates.insert({cut_cells,
          // contact_candidates});
        }
    }
}


template class ParticleWallBroadSearch<2>;
template class ParticleWallBroadSearch<3>;
