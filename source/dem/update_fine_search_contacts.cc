#include <dem/particle_particle_contact_info_struct.h>
#include <dem/update_fine_search_contacts.h>

using namespace dealii;

template <int dim>
void
update_particle_fine_search_candidates(
  typename dem_data_containers::dem_data_structures<
    dim>::adjacent_particle_pairs &adjacent_particles,
  typename dem_data_containers::dem_data_structures<
    dim>::particle_particle_candidates &contact_pair_candidates,
  const std::string                     check_type)
{
  // Loop over all the local particles which have local adjacent particles
  for (auto adjacent_particles_iterator = adjacent_particles.begin();
       adjacent_particles_iterator != adjacent_particles.end();
       ++adjacent_particles_iterator)
    {
      // Get the current particle id
      auto particle_one_id = adjacent_particles_iterator->first;

      // Get the reference to all the particle contact candidates for the
      // current particle (in neighbor cells, from the board search)
      auto particle_one_contact_candidates =
        &contact_pair_candidates[particle_one_id];

      // Get adjacent particles content of the current particle
      // (in the threshold of the fine search of the particle)
      auto adjacent_pairs_content = &adjacent_particles_iterator->second;

      // Loop over all the adjacent particles
      for (auto particle_particle_map_iterator =
             adjacent_pairs_content->begin();
           particle_particle_map_iterator != adjacent_pairs_content->end();)
        {
          // Get the current local adjacent particle id and the reference to
          // all the local particle contact candidates of this particle
          auto particle_two_id = particle_particle_map_iterator->first;
          auto particle_two_contact_candidates =
            &contact_pair_candidates[particle_two_id];

          // Check if the current local adjacent particle is a
          // particle candidate of the current local particle
          auto search_iterator_one =
            std::find(particle_one_contact_candidates->begin(),
                      particle_one_contact_candidates->end(),
                      particle_two_id);

          // Check if the current local particle is a local particle candidate
          // of the current local adjacent particle
          auto search_iterator_two =
            std::find(particle_two_contact_candidates->begin(),
                      particle_two_contact_candidates->end(),
                      particle_one_id);

          // If the current local adjacent particle is a candidate to current
          // local particle, it is removed from current local particle candidate
          // list and vice versa.
          if (search_iterator_one != particle_one_contact_candidates->end())
            {
              particle_one_contact_candidates->erase(search_iterator_one);
              ++particle_particle_map_iterator;
            }
          else if (search_iterator_two !=
                     particle_two_contact_candidates->end() &&
                   check_type == "local")
            {
              particle_two_contact_candidates->erase(search_iterator_two);
              ++particle_particle_map_iterator;
            }
          else if (search_iterator_two !=
                     particle_two_contact_candidates->end() &&
                   check_type == "ghost")
            {
              adjacent_pairs_content->erase(particle_particle_map_iterator++);
            }
          else
            {
              adjacent_pairs_content->erase(particle_particle_map_iterator++);
            }
        }
    }
}



template <int dim, typename structure>
void
update_wall_fine_search_candidates(
  typename dem_data_containers::dem_data_structures<
    dim>::particle_wall_in_contact &particle_wall_pairs_in_contact,
  structure &                       particle_wall_contact_candidates)
{
  for (auto particle_wall_pairs_in_contact_iterator =
         particle_wall_pairs_in_contact.begin();
       particle_wall_pairs_in_contact_iterator !=
       particle_wall_pairs_in_contact.end();
       ++particle_wall_pairs_in_contact_iterator)
    {
      auto particle_id = particle_wall_pairs_in_contact_iterator->first;

      auto adjacent_pairs_content =
        &particle_wall_pairs_in_contact_iterator->second;

      for (auto particle_wall_map_iterator = adjacent_pairs_content->begin();
           particle_wall_map_iterator != adjacent_pairs_content->end();)
        {
          auto wall_id = particle_wall_map_iterator->first;
          auto particle_wall_contact_candidate_element =
            &particle_wall_contact_candidates[particle_id];

          auto search_iterator =
            particle_wall_contact_candidate_element->find(wall_id);

          if (search_iterator != particle_wall_contact_candidate_element->end())
            {
              particle_wall_contact_candidate_element->erase(search_iterator);
              ++particle_wall_map_iterator;
            }
          else
            {
              adjacent_pairs_content->erase(particle_wall_map_iterator++);
            }
        }
    }
}

template <int dim>
void
update_mesh_fine_search_candidates(
  typename dem_data_containers::dem_data_structures<
    dim>::particle_floating_mesh_in_contact &particle_floating_mesh_in_contact,
  typename dem_data_containers::dem_data_structures<
    dim>::particle_floating_mesh_candidates
    &particle_floating_mesh_contact_candidates)
{
  for (unsigned int solid_counter = 0;
       solid_counter < particle_floating_mesh_in_contact.size();
       ++solid_counter)
    {
      auto &particle_floating_mesh_contact_pair =
        particle_floating_mesh_in_contact[solid_counter];

      for (auto particle_floating_mesh_pairs_in_contact_iterator =
             particle_floating_mesh_contact_pair.begin();
           particle_floating_mesh_pairs_in_contact_iterator !=
           particle_floating_mesh_contact_pair.end();
           ++particle_floating_mesh_pairs_in_contact_iterator)
        {
          auto triangle =
            particle_floating_mesh_pairs_in_contact_iterator->first;

          auto adjacent_pairs_content =
            &particle_floating_mesh_pairs_in_contact_iterator->second;

          for (auto particle_floating_mesh_map_iterator =
                 adjacent_pairs_content->begin();
               particle_floating_mesh_map_iterator !=
               adjacent_pairs_content->end();)
            {
              auto particle_id = particle_floating_mesh_map_iterator->first;

              auto particle_floating_mesh_candidate_element =
                &particle_floating_mesh_contact_candidates[solid_counter]
                                                          [triangle];

              auto search_iterator =
                particle_floating_mesh_candidate_element->find(particle_id);

              if (search_iterator !=
                  particle_floating_mesh_candidate_element->end())
                {
                  particle_floating_mesh_candidate_element->erase(
                    search_iterator);
                  ++particle_floating_mesh_map_iterator;
                }
              else
                {
                  adjacent_pairs_content->erase(
                    particle_floating_mesh_map_iterator++);
                }
            }
        }
    }
}

template void update_particle_fine_search_candidates<2>(
  typename dem_data_containers::dem_data_structures<2>::adjacent_particle_pairs
    &adjacent_particles,
  typename dem_data_containers::dem_data_structures<
    2>::particle_particle_candidates &contact_pair_candidates,
  const std::string                   check_type);

template void update_particle_fine_search_candidates<3>(
  typename dem_data_containers::dem_data_structures<3>::adjacent_particle_pairs
    &adjacent_particles,
  typename dem_data_containers::dem_data_structures<
    3>::particle_particle_candidates &contact_pair_candidates,
  const std::string                   check_type);

template void update_wall_fine_search_candidates<
  2,
  typename dem_data_containers::dem_data_structures<
    2>::particle_wall_candidates>(
  typename dem_data_containers::dem_data_structures<2>::particle_wall_in_contact
    &particle_wall_pairs_in_contact,
  typename dem_data_containers::dem_data_structures<2>::particle_wall_candidates
    &particle_wall_contact_candidates);

template void update_wall_fine_search_candidates<
  3,
  typename dem_data_containers::dem_data_structures<
    3>::particle_wall_candidates>(
  typename dem_data_containers::dem_data_structures<3>::particle_wall_in_contact
    &particle_wall_pairs_in_contact,
  typename dem_data_containers::dem_data_structures<3>::particle_wall_candidates
    &particle_wall_contact_candidates);

template void update_wall_fine_search_candidates<
  2,
  typename dem_data_containers::dem_data_structures<
    2>::particle_floating_wall_candidates>(
  typename dem_data_containers::dem_data_structures<2>::particle_wall_in_contact
    &particle_wall_pairs_in_contact,
  typename dem_data_containers::dem_data_structures<
    2>::particle_floating_wall_candidates &particle_wall_contact_candidates);

template void update_wall_fine_search_candidates<
  3,
  typename dem_data_containers::dem_data_structures<
    3>::particle_floating_wall_candidates>(
  typename dem_data_containers::dem_data_structures<3>::particle_wall_in_contact
    &particle_wall_pairs_in_contact,
  typename dem_data_containers::dem_data_structures<
    3>::particle_floating_wall_candidates &particle_wall_contact_candidates);

template void update_mesh_fine_search_candidates<2>(
  typename dem_data_containers::dem_data_structures<
    2>::particle_floating_mesh_in_contact &particle_floating_mesh_in_contact,
  typename dem_data_containers::dem_data_structures<
    2>::particle_floating_mesh_candidates
    &particle_floating_mesh_contact_candidates);

template void update_mesh_fine_search_candidates<3>(
  typename dem_data_containers::dem_data_structures<
    3>::particle_floating_mesh_in_contact &particle_floating_mesh_in_contact,
  typename dem_data_containers::dem_data_structures<
    3>::particle_floating_mesh_candidates
    &particle_floating_mesh_contact_candidates);