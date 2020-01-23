#include <dem/explicit_euler_integrator.h>

template <int dim, int spacedim>
void
ExplicitEulerIntegrator<dim, spacedim>::integrate(
  Particles::ParticleHandler<dim, spacedim> &particle_handler,
  Tensor<1, dim>                             g,
  double                                     dt)
{
  for (auto particle = particle_handler.begin();
       particle != particle_handler.end();
       ++particle)
    {
      // Get the total array view to the particle properties once to improve
      // efficiency
      auto particle_properties = particle->get_properties();

      // Acceleration calculation:
      particle_properties[10] =
        g[0] + particle_properties[13] / particle_properties[19];
      particle_properties[11] =
        g[1] + particle_properties[14] / particle_properties[19];
      particle_properties[12] =
        g[2] + particle_properties[15] / particle_properties[19];

      // Velocity integration:
      Tensor<1, dim> particle_velocity;
      particle_velocity[0] =
        particle_properties[7] + dt * particle_properties[10];
      particle_velocity[1] =
        particle_properties[8] + dt * particle_properties[11];
      particle_velocity[2] =
        particle_properties[9] + dt * particle_properties[12];
      for (unsigned int i = 0; i < dim; ++i)
        {
          particle_properties[7 + i] = particle_velocity[i];
        }

      // Position integration:
      auto particle_position = particle->get_location();
      particle_position      = particle_position + (particle_velocity * dt);
      particle->set_location(particle_position);

      // Angular velocity:
      /*
      particle->get_properties()[16] = particle->get_properties()[16] +
        (particle->get_properties()[21]) / (particle->get_properties()[20]);
      particle->get_properties()[17] =particle->get_properties()[17] +
        (particle->get_properties()[22]) / (particle->get_properties()[20]);
      particle->get_properties()[18] = particle->get_properties()[18] +
        (particle->get_properties()[23]) / (particle->get_properties()[20]);
        */
    }
}

template class ExplicitEulerIntegrator<3, 3>;
