#include <dem/locate_local_particles.h>

using namespace dealii;

template <int dim>
void
locate_local_particles_in_cells(
  const Particles::ParticleHandler<dim> &particle_handler,
  std::unordered_map<types::particle_index, Particles::ParticleIterator<dim>>
    &particle_container,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index, pp_contact_info_struct<dim>>>
    &ghost_adjacent_particles,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index, pp_contact_info_struct<dim>>>
    &local_adjacent_particles,
  std::unordered_map<
    types::particle_index,
    std::map<types::particle_index, pw_contact_info_struct<dim>>>
    &pw_pairs_in_contact,
  std::unordered_map<
    types::particle_index,
    std::map<types::particle_index, pw_contact_info_struct<dim>>>
    &pfw_pairs_in_contact,
  std::unordered_map<types::particle_index,
                     particle_point_line_contact_info_struct<dim>>
    &particle_points_in_contact,
  std::unordered_map<types::particle_index,
                     particle_point_line_contact_info_struct<dim>>
    &particle_lines_in_contact)
{
  update_particle_container<dim>(particle_container, &particle_handler);

  update_local_pp_contact_container_iterators<dim>(local_adjacent_particles,
                                                   particle_container);

  update_ghost_pp_contact_container_iterators<dim>(ghost_adjacent_particles,
                                                   particle_container);

  update_pw_contact_container_iterators<dim>(pw_pairs_in_contact,
                                             particle_container);

  // Calling the same function for floating walls
  update_pw_contact_container_iterators<dim>(pfw_pairs_in_contact,
                                             particle_container);

  update_particle_point_line_contact_container_iterators<dim>(
    particle_points_in_contact, particle_lines_in_contact, particle_container);
}

template void
locate_local_particles_in_cells(
  const Particles::ParticleHandler<2> &particle_handler,
  std::unordered_map<types::particle_index, Particles::ParticleIterator<2>>
    &particle_container,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index, pp_contact_info_struct<2>>>
    &ghost_adjacent_particles,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index, pp_contact_info_struct<2>>>
    &local_adjacent_particles,
  std::unordered_map<types::particle_index,
                     std::map<types::particle_index, pw_contact_info_struct<2>>>
    &pw_pairs_in_contact,
  std::unordered_map<types::particle_index,
                     std::map<types::particle_index, pw_contact_info_struct<2>>>
    &pfw_pairs_in_contact,
  std::unordered_map<types::particle_index,
                     particle_point_line_contact_info_struct<2>>
    &particle_points_in_contact,
  std::unordered_map<types::particle_index,
                     particle_point_line_contact_info_struct<2>>
    &particle_lines_in_contact);

template void
locate_local_particles_in_cells(
  const Particles::ParticleHandler<3> &particle_handler,
  std::unordered_map<types::particle_index, Particles::ParticleIterator<3>>
    &particle_container,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index, pp_contact_info_struct<3>>>
    &ghost_adjacent_particles,
  std::unordered_map<
    types::particle_index,
    std::unordered_map<types::particle_index, pp_contact_info_struct<3>>>
    &local_adjacent_particles,
  std::unordered_map<types::particle_index,
                     std::map<types::particle_index, pw_contact_info_struct<3>>>
    &pw_pairs_in_contact,
  std::unordered_map<types::particle_index,
                     std::map<types::particle_index, pw_contact_info_struct<3>>>
    &pfw_pairs_in_contact,
  std::unordered_map<types::particle_index,
                     particle_point_line_contact_info_struct<3>>
    &particle_points_in_contact,
  std::unordered_map<types::particle_index,
                     particle_point_line_contact_info_struct<3>>
    &particle_lines_in_contact);
