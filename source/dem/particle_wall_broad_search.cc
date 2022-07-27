
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
          std::unordered_map<types::boundary_id,
                             std::tuple<Particles::ParticleIterator<dim>,
                                        Tensor<1, dim>,
                                        Point<dim>,
                                        types::boundary_id,
                                        types::global_cell_index>>>
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
    std::unordered_map<types::boundary_id, Particles::ParticleIterator<dim>>>
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
      auto floating_wall_id =
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
ParticleWallBroadSearch<dim>::particle_floating_mesh_contact_search(
  const std::vector<std::vector<
    std::pair<typename Triangulation<dim>::active_cell_iterator,
              typename Triangulation<dim - 1, dim>::active_cell_iterator>>>
    &                                    floating_mesh_information,
  const Particles::ParticleHandler<dim> &particle_handler,
  std::vector<std::map<
    typename Triangulation<dim - 1, dim>::active_cell_iterator,
    std::unordered_map<types::particle_index, Particles::ParticleIterator<dim>>,
    dem_data_containers::cut_cell_comparison<dim>>>
    &particle_floating_mesh_contact_candidates,
  std::unordered_map<
    types::global_cell_index,
    std::vector<typename Triangulation<dim>::active_cell_iterator>>
    &cells_total_neighbor_list)
{
  // Clear the candidate container
  particle_floating_mesh_contact_candidates.clear();
  particle_floating_mesh_contact_candidates.resize(
    floating_mesh_information.size());

  for (unsigned int solid_counter = 0;
       solid_counter < floating_mesh_information.size();
       ++solid_counter)
    {
      auto &candidates =
        particle_floating_mesh_contact_candidates[solid_counter];

      // I am assuming that triangles in different solids have different unique
      // global ids. If it's not the case, we have to modify the code

      // Loop through solids
      auto &solid_iterator = floating_mesh_information[solid_counter];

      // Loop through the pairs (first -> background cell, second -> floating
      // cell)
      for (auto floating_mesh_iterator = solid_iterator.begin();
           floating_mesh_iterator != solid_iterator.end();
           ++floating_mesh_iterator)
        {
          // Get background cell
          auto background_cell = floating_mesh_iterator->first;

          if (background_cell->is_locally_owned())
            {
              // Get cut cells (floating mesh cells)
              auto cut_cells = floating_mesh_iterator->second;

              // Get neighbors of the background cell
              auto cell_list = cells_total_neighbor_list.at(
                background_cell->global_active_cell_index());

              // Loop through neighbors
              for (auto &cell_iterator : cell_list)
                {
                  // Find particles located in cell
                  typename Particles::ParticleHandler<
                    dim>::particle_iterator_range particles_in_cell =
                    particle_handler.particles_in_cell(cell_iterator);

                  const bool particles_exist_in_cell =
                    !particles_in_cell.empty();

                  // If the main cell is not empty
                  if (particles_exist_in_cell)
                    {
                      // Loop through particles in the main cell and build
                      // contact candidate pairs
                      for (typename Particles::ParticleHandler<
                             dim>::particle_iterator_range::iterator
                             particles_in_cell_iterator =
                               particles_in_cell.begin();
                           particles_in_cell_iterator !=
                           particles_in_cell.end();
                           ++particles_in_cell_iterator)
                        {
                          candidates[cut_cells].insert(
                            {particles_in_cell_iterator->get_id(),
                             particles_in_cell_iterator});
                        }
                    }
                }
            }
        }
    }
}


template class ParticleWallBroadSearch<2>;
template class ParticleWallBroadSearch<3>;
