/* ---------------------------------------------------------------------
*
* Copyright (C) 2019 - 2022 by the Lethe authors
*
* This file is part of the Lethe library
*
* The Lethe library is free software; you can use it, redistribute
* it, and/or modify it under the terms of the GNU Lesser General
* Public License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
* The full text of the license can be found in the file LICENSE at
* the top level of the Lethe distribution.
*
* ---------------------------------------------------------------------

*
*/

#include <dem/disable_particle_contact.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

template <int dim>
DisableParticleContact<dim>::DisableParticleContact()
  : status_to_cell(mobility_status::n_mobility_status)
{}

template <int dim>
void
DisableParticleContact<dim>::calculate_average_granular_temperature(
  const DoFHandler<dim> &                background_dh,
  const Particles::ParticleHandler<dim> &particle_handler)
{
  auto triangulation = &background_dh.get_triangulation();
  granular_temperature_average.reinit(triangulation->n_active_cells());
  solid_fractions.reinit(triangulation->n_active_cells());
  active_ghost_cells.clear();

  // Iterating through the active cells in the triangulation
  for (const auto &cell : background_dh.active_cell_iterators())
    {
      if (cell->is_locally_owned() || cell->is_ghost())
        {
          // Get active and ghost cells in a set
          active_ghost_cells.insert(cell);
          double granular_temperature_cell = 0.0;
          double solid_fraction            = 0.0;

          // Particles in the cell
          typename Particles::ParticleHandler<dim>::particle_iterator_range
                             particles_in_cell = particle_handler.particles_in_cell(cell);
          const unsigned int n_particles_in_cell =
            particle_handler.n_particles_in_cell(cell);

          // Check if the cell has any particles
          if (n_particles_in_cell > 0)
            {
              // Initialize variables for average velocity
              Tensor<1, dim> velocity_cell_sum;
              Tensor<1, dim> velocity_cell_average;

              // Initialize variables for void fraction
              double       solid_volume = 0.0;
              const double cell_volume  = cell->measure();

              // Initialize velocity fluctuations
              Tensor<1, dim> cell_velocity_fluctuation_squared_sum;
              Tensor<1, dim> cell_velocity_fluctuation_squared_average;

              // First loop over particles in cell to calculation the average
              // velocity and the void fraction
              for (typename Particles::ParticleHandler<
                     dim>::particle_iterator_range::iterator
                     particles_in_cell_iterator = particles_in_cell.begin();
                   particles_in_cell_iterator != particles_in_cell.end();
                   ++particles_in_cell_iterator)
                {
                  auto &particle_properties =
                    particles_in_cell_iterator->get_properties();

                  for (int d = 0; d < dim; ++d)
                    {
                      velocity_cell_sum[d] +=
                        particle_properties[DEM::PropertiesIndex::v_x + d];
                    }

                  solid_volume +=
                    M_PI *
                    pow(particle_properties[DEM::PropertiesIndex::dp], dim) /
                    (2.0 * dim);
                }

              // Calculate average velocity in the cell
              for (int d = 0; d < dim; ++d)
                velocity_cell_average[d] =
                  velocity_cell_sum[d] / n_particles_in_cell;

              // Calculate void fraction of cell
              solid_fraction = solid_volume / cell_volume;

              // Second loop over particle to calculate the average granular
              // temperature
              for (typename Particles::ParticleHandler<
                     dim>::particle_iterator_range::iterator
                     particles_in_cell_iterator = particles_in_cell.begin();
                   particles_in_cell_iterator != particles_in_cell.end();
                   ++particles_in_cell_iterator)
                {
                  auto &particle_properties =
                    particles_in_cell_iterator->get_properties();

                  for (int d = 0; d < dim; ++d)
                    {
                      cell_velocity_fluctuation_squared_sum[d] +=
                        (particle_properties[DEM::PropertiesIndex::v_x + d] -
                         velocity_cell_average[d]) *
                        (particle_properties[DEM::PropertiesIndex::v_x + d] -
                         velocity_cell_average[d]);
                    }
                }

              // Calculate average granular temperature in the cell
              for (int d = 0; d < dim; ++d)
                {
                  cell_velocity_fluctuation_squared_average[d] =
                    cell_velocity_fluctuation_squared_sum[d] /
                    n_particles_in_cell;
                  granular_temperature_cell +=
                    (1.0 / dim) * cell_velocity_fluctuation_squared_average[d];
                }
            }

          // Store the average granular temperature and solid fraction with
          // active cell index
          granular_temperature_average[cell->active_cell_index()] =
            granular_temperature_cell;
          solid_fractions[cell->active_cell_index()] = solid_fraction;
        }
    }
}


template <int dim>
void
DisableParticleContact<dim>::identify_mobility_status(
  const DoFHandler<dim> &                background_dh,
  const Particles::ParticleHandler<dim> &particle_handler,
  MPI_Comm                               mpi_communicator)
{
  // Reset cell status containers
  status_to_cell.clear();
  status_to_cell.resize(mobility_status::n_mobility_status);

  // Create dummy dofs for background dof handler
  const FE_Q<dim>    fe(1);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  // Get locally owned and relevant dofs
  const IndexSet locally_owned_dofs = background_dh.locally_owned_dofs();
  IndexSet       locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(background_dh);

  // Reinit all value of mobility at nodes
  mobility_at_nodes.reinit(locally_owned_dofs,
                           locally_relevant_dofs,
                           mpi_communicator);
  mobility_at_nodes = 0;

  // Empty status (3) to nodes when no particle in cell
  for (auto cell = active_ghost_cells.begin();
       cell != active_ghost_cells.end();)
    {
      // Check if the cell has any particles
      if (particle_handler.n_particles_in_cell(*cell) == 0)
        {
          std::vector<types::global_dof_index> local_dofs_indices(
            dofs_per_cell);
          (*cell)->get_dof_indices(local_dofs_indices);

          // Remove empty cell from cells to reduce the number of check
          cell = active_ghost_cells.erase(cell);

          // Assign empty status to nodes
          for (auto dof_index : local_dofs_indices)
            {
              mobility_at_nodes(dof_index) = mobility_status::empty;
            }
        }
      else
        {
          ++cell;
        }
    }

  mobility_at_nodes.update_ghost_values();

  // Mobile status (2) to nodes (no overwrite of empty status) and to cell if
  // the criteria is respected or cell has one or many empty nodes
  // (empty neighbor)
  for (auto cell = active_ghost_cells.begin();
       cell != active_ghost_cells.end();)
    {
      // Assign mobility status to cell if has particles and
      // - granular temperature > limit or
      // - solid fraction of cell < limit or
      // - is next to an empty cell
      const unsigned int cell_id = (*cell)->active_cell_index();

      std::vector<types::global_dof_index> local_dofs_indices(dofs_per_cell);
      (*cell)->get_dof_indices(local_dofs_indices);

      // Check if cell has an empty status node
      bool has_empty_neighbor = false;
      for (auto dof_index : local_dofs_indices)
        {
          if (mobility_at_nodes[dof_index] == mobility_status::empty)
            {
              has_empty_neighbor = true;
              break;
            }
        }

      if (granular_temperature_average[cell_id] >= granular_temperature_limit ||
          solid_fractions[cell_id] <= solid_fraction_limit ||
          has_empty_neighbor)
        {
          // Insert cell in mobile status set
          status_to_cell[mobility_status::mobile].insert(*cell);

          // Remove cell from cell set
          cell = active_ghost_cells.erase(cell);

          // Assign mobile status to nodes
          for (auto dof_index : local_dofs_indices)
            {
              // Don't overwrite empty nodes
              mobility_at_nodes(dof_index) =
                std::max((int)mobility_status::mobile,
                         mobility_at_nodes[dof_index]);
            }
        }
      else
        {
          ++cell;
        }
    }

  mobility_at_nodes.update_ghost_values();

  // Layer of mobile cells over mobile cells
  for (auto cell = active_ghost_cells.begin();
       cell != active_ghost_cells.end();)
    {
      std::vector<types::global_dof_index> local_dofs_indices(dofs_per_cell);
      (*cell)->get_dof_indices(local_dofs_indices);
      bool has_mobile_nodes = false;

      // Loop over nodes of cell
      for (auto dof_index : local_dofs_indices)
        {
          // Check if node is mobile and assign mobile status to
          // the cell
          if (mobility_at_nodes[dof_index] == mobility_status::mobile)
            {
              status_to_cell[mobility_status::mobile].insert(*cell);

              // Remove cell from cell set
              cell             = active_ghost_cells.erase(cell);
              has_mobile_nodes = true;

              // Assign active status to nodes except mobile because
              // this will cause to propagate the mobile status to the
              // neighbors in this loop since the mobility check at node
              // is executed in the same container that we assign new
              // mobility status
              // Since mobile = 2 > active = 1, this step won't overwrite
              // mobile nodes
              for (auto dof_index : local_dofs_indices)
                {
                  mobility_at_nodes[dof_index] =
                    std::max((int)mobility_status::active,
                             mobility_at_nodes[dof_index]);
                }
              break;
            }
        }

      if (!has_mobile_nodes)
        {
          ++cell;
        }
    }

  mobility_at_nodes.update_ghost_values();

  // Layer of neighbor of mobile cells
  for (auto cell = active_ghost_cells.begin();
       cell != active_ghost_cells.end();)
    {
      std::vector<types::global_dof_index> local_dofs_indices(dofs_per_cell);
      (*cell)->get_dof_indices(local_dofs_indices);

      bool has_active_nodes = false;
      bool has_mobile_nodes = false;

      // Loop over nodes of cell and check if cell has active or mobile
      // nodes
      // Active nodes and no mobiles nodes means that the cell is a neighbor
      // of the layer of cell over mobily cell by criterion Active nodes
      // with mobiles nodes means that the cell is part of this layer
      for (auto dof_index : local_dofs_indices)
        {
          has_active_nodes =
            (mobility_at_nodes[dof_index] == mobility_status::active) ||
            has_active_nodes;

          has_mobile_nodes =
            (mobility_at_nodes[dof_index] == mobility_status::mobile) ||
            has_mobile_nodes;
        }

      if (has_active_nodes && !has_mobile_nodes)
        {
          status_to_cell[mobility_status::active].insert(*cell);
        }

      ++cell;
    }
}

template class DisableParticleContact<2>;
template class DisableParticleContact<3>;