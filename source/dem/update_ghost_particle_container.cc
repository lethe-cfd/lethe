#include <dem/update_ghost_particle_container.h>

using namespace dealii;

template <int dim>
void
update_ghost_particle_container(
  std::unordered_map<types::particle_index, Particles::ParticleIterator<dim>>
    &                                    ghost_particle_container,
  const Particles::ParticleHandler<dim> *particle_handler)
{
  ghost_particle_container.clear();

  for (auto particle_iterator = particle_handler->begin_ghost();
       particle_iterator != particle_handler->end_ghost();
       ++particle_iterator)
    {
      ghost_particle_container[particle_iterator->get_id()] = particle_iterator;
    }
}

template void update_ghost_particle_container(
  std::unordered_map<types::particle_index, Particles::ParticleIterator<2>>
    &                                  ghost_particle_container,
  const Particles::ParticleHandler<2> *particle_handler);

template void update_ghost_particle_container(
  std::unordered_map<types::particle_index, Particles::ParticleIterator<3>>
    &                                  ghost_particle_container,
  const Particles::ParticleHandler<3> *particle_handler);
