#include <solvers/navier_stokes_assemblers.h>

template <int dim>
void
GLSNavierStokesAssemblerCore<dim>::assemble_matrix(
  NavierStokesScratchData<dim> &        scratch_data,
  StabilizedMethodsTensorCopyData<dim> &copy_data)
{
  // Scheme and physical properties
  const double viscosity = scratch_data.physical_properties.viscosity;

  // Loop and quadrature informations
  const auto &       JxW_vec    = scratch_data.JxW;
  const unsigned int n_q_points = scratch_data.n_q_points;
  const unsigned int n_dofs     = scratch_data.n_dofs;
  const double       h          = scratch_data.cell_size;

  // Copy data elements
  auto &strong_residual_vec = copy_data.strong_residual;
  auto &strong_jacobian_vec = copy_data.strong_jacobian;
  auto &local_matrix        = copy_data.local_matrix;


  // Loop over the quadrature points
  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      // Gather into local variables the relevant fields
      const std::vector<Tensor<1, dim>> velocity = {
        scratch_data.velocity_values[q]};
      const Tensor<2, dim> velocity_gradient =
        scratch_data.velocity_gradients[q];
      const Tensor<1, dim> velocity_laplacian =
        scratch_data.velocity_laplacians[q];

      const Tensor<1, dim> pressure_gradient =
        scratch_data.pressure_gradients[q];

      // Forcing term
      const Tensor<1, dim> force = scratch_data.force[q];

      // Calculation of the magnitude of the velocity for the
      // stabilization parameter
      const double u_mag = std::max(velocity[0].norm(), 1e-12);

      // Store JxW in local variable for faster access;
      const double JxW = JxW_vec[q];

      // Calculation of the GLS stabilization parameter. The
      // stabilization parameter used is different if the simulation
      // is steady or unsteady. In the unsteady case it includes the
      // value of the time-step
      // BB-TODO fix transient character
      const double tau =
        true ?
          1. / std::sqrt(std::pow(2. * u_mag / h, 2) +
                         9 * std::pow(4 * viscosity / (h * h), 2)) :
          1. / std::sqrt(std::pow(1 /*sdt*/, 2) + std::pow(2. * u_mag / h, 2) +
                         9 * std::pow(4 * viscosity / (h * h), 2));

      // Calculate the strong residual for GLS stabilization
      auto strong_residual =
        velocity_gradient * velocity[0] + pressure_gradient -
        viscosity * velocity_laplacian - force + strong_residual_vec[q];

      std::vector<Tensor<1, dim>> grad_phi_u_j_x_velocity(n_dofs);
      std::vector<Tensor<1, dim>> velocity_gradient_x_phi_u_j(n_dofs);


      // We loop over the column first to prevent recalculation
      // of the strong jacobian in the inner loop
      for (unsigned int j = 0; j < n_dofs; ++j)
        {
          const auto &phi_u_j           = scratch_data.phi_u[q][j];
          const auto &grad_phi_u_j      = scratch_data.grad_phi_u[q][j];
          const auto &laplacian_phi_u_j = scratch_data.laplacian_phi_u[q][j];

          const auto &grad_phi_p_j = scratch_data.grad_phi_p[q][j];

          strong_jacobian_vec[q][j] +=
            (velocity_gradient * phi_u_j + grad_phi_u_j * velocity[0] +
             grad_phi_p_j - viscosity * laplacian_phi_u_j);

          // Store these temporary products in auxiliary variables for speed
          grad_phi_u_j_x_velocity[j]     = grad_phi_u_j * velocity[0];
          velocity_gradient_x_phi_u_j[j] = velocity_gradient * phi_u_j;
        }



      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const auto &phi_u_i      = scratch_data.phi_u[q][i];
          const auto &grad_phi_u_i = scratch_data.grad_phi_u[q][i];
          const auto &div_phi_u_i  = scratch_data.div_phi_u[q][i];
          const auto &phi_p_i      = scratch_data.phi_p[q][i];
          const auto &grad_phi_p_i = scratch_data.grad_phi_p[q][i];

          // Store these temporary products in auxiliary variables for speed
          const auto grad_phi_u_i_x_velocity = grad_phi_u_i * velocity[0];
          const auto strong_residual_x_grad_phi_u_i =
            strong_residual * grad_phi_u_i;

          for (unsigned int j = 0; j < n_dofs; ++j)
            {
              const auto &phi_u_j      = scratch_data.phi_u[q][j];
              const auto &grad_phi_u_j = scratch_data.grad_phi_u[q][j];
              const auto &div_phi_u_j  = scratch_data.div_phi_u[q][j];

              const auto &phi_p_j = scratch_data.phi_p[q][j];

              const auto &strong_jac = strong_jacobian_vec[q][j];

              double local_matrix_ij =
                viscosity * scalar_product(grad_phi_u_j, grad_phi_u_i) +
                velocity_gradient_x_phi_u_j[j] * phi_u_i +
                grad_phi_u_j_x_velocity[j] * phi_u_i - div_phi_u_i * phi_p_j +
                // Continuity
                phi_p_i * div_phi_u_j;

              // PSPG GLS term
              local_matrix_ij += tau * (strong_jac * grad_phi_p_i);


              // Jacobian is currently incomplete
              if (SUPG)
                {
                  local_matrix_ij +=
                    tau * (strong_jac * grad_phi_u_i_x_velocity +
                           strong_residual_x_grad_phi_u_i * phi_u_j);
                }
              local_matrix_ij *= JxW;
              local_matrix(i, j) += local_matrix_ij;
            }
        }
    }
}

template <int dim>
void
GLSNavierStokesAssemblerCore<dim>::assemble_rhs(
  NavierStokesScratchData<dim> &        scratch_data,
  StabilizedMethodsTensorCopyData<dim> &copy_data)
{
  // Scheme and physical properties
  const double viscosity = scratch_data.physical_properties.viscosity;

  // Loop and quadrature informations
  const auto &       JxW_vec    = scratch_data.JxW;
  const unsigned int n_q_points = scratch_data.n_q_points;
  const unsigned int n_dofs     = scratch_data.n_dofs;
  const double       h          = scratch_data.cell_size;

  // Copy data elements
  auto &strong_residual_vec = copy_data.strong_residual;
  auto &local_rhs           = copy_data.local_rhs;


  // Loop over the quadrature points
  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      // Velocity
      const std::vector<Tensor<1, dim>> velocity = {
        scratch_data.velocity_values[q]};
      const double velocity_divergence = scratch_data.velocity_divergences[q];
      const Tensor<2, dim> velocity_gradient =
        scratch_data.velocity_gradients[q];
      const Tensor<1, dim> velocity_laplacian =
        scratch_data.velocity_laplacians[q];

      // Pressure
      const double         pressure = scratch_data.pressure_values[q];
      const Tensor<1, dim> pressure_gradient =
        scratch_data.pressure_gradients[q];

      // Forcing term
      const Tensor<1, dim> force = scratch_data.force[q];

      // Calculation of the magnitude of the
      // velocity for the stabilization parameter
      const double u_mag = std::max(velocity[0].norm(), 1e-12);

      // Store JxW in local variable for faster access;
      const double JxW = JxW_vec[q];

      // Calculation of the GLS stabilization parameter. The
      // stabilization parameter used is different if the simulation
      // is steady or unsteady. In the unsteady case it includes the
      // value of the time-step
      // BB-TODO fix transient character
      const double tau =
        true ?
          1. / std::sqrt(std::pow(2. * u_mag / h, 2) +
                         9 * std::pow(4 * viscosity / (h * h), 2)) :
          1. / std::sqrt(std::pow(1 /*sdt*/, 2) + std::pow(2. * u_mag / h, 2) +
                         9 * std::pow(4 * viscosity / (h * h), 2));


      // Calculate the strong residual for GLS stabilization
      auto strong_residual =
        velocity_gradient * velocity[0] + pressure_gradient -
        viscosity * velocity_laplacian - force + strong_residual_vec[q];

      // Assembly of the right-hand side
      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const auto phi_u_i      = scratch_data.phi_u[q][i];
          const auto grad_phi_u_i = scratch_data.grad_phi_u[q][i];
          const auto phi_p_i      = scratch_data.phi_p[q][i];
          const auto grad_phi_p_i = scratch_data.grad_phi_p[q][i];
          const auto div_phi_u_i  = scratch_data.div_phi_u[q][i];


          // Navier-Stokes Residual
          local_rhs(i) +=
            (
              // Momentum
              -viscosity * scalar_product(velocity_gradient, grad_phi_u_i) -
              velocity_gradient * velocity[0] * phi_u_i +
              pressure * div_phi_u_i + force * phi_u_i -
              // Continuity
              velocity_divergence * phi_p_i) *
            JxW;

          // PSPG GLS term
          local_rhs(i) += -tau * (strong_residual * grad_phi_p_i) * JxW;

          // SUPG GLS term
          if (SUPG)
            {
              local_rhs(i) +=
                -tau * (strong_residual * (grad_phi_u_i * velocity[0])) * JxW;
            }
        }
    }
}



template class GLSNavierStokesAssemblerCore<2>;
template class GLSNavierStokesAssemblerCore<3>;


template <int dim>
void
GLSNavierStokesAssemblerSRF<dim>::assemble_matrix(
  NavierStokesScratchData<dim> &        scratch_data,
  StabilizedMethodsTensorCopyData<dim> &copy_data)
{
  // Loop and quadrature informations
  const auto &       JxW        = scratch_data.JxW;
  const unsigned int n_q_points = scratch_data.n_q_points;
  const unsigned int n_dofs     = scratch_data.n_dofs;

  // Copy data elements
  auto &strong_residual = copy_data.strong_residual;
  auto &strong_jacobian = copy_data.strong_jacobian;
  auto &local_matrix    = copy_data.local_matrix;

  // SRF Source term
  //----------------------------------
  // Angular velocity of the rotating frame. This is always a 3D vector even
  // in 2D.
  Tensor<1, dim> omega_vector;

  double omega_z  = velocity_sources.omega_z;
  omega_vector[0] = velocity_sources.omega_x;
  omega_vector[1] = velocity_sources.omega_y;
  if (dim == 3)
    omega_vector[2] = velocity_sources.omega_z;


  // Loop over the quadrature points
  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      // Velocity
      const std::vector<Tensor<1, dim>> velocity = {
        scratch_data.velocity_values[q]};

      if (dim == 2)
        {
          strong_residual[q] +=
            2 * omega_z * (-1.) * cross_product_2d(velocity[0]);
          auto centrifugal =
            omega_z * (-1.) *
            cross_product_2d(
              omega_z * (-1.) *
              cross_product_2d(scratch_data.quadrature_points[q]));
          strong_residual[q] += centrifugal;
        }
      else // dim == 3
        {
          strong_residual[q] += 2 * cross_product_3d(omega_vector, velocity[0]);
          strong_residual[q] += cross_product_3d(
            omega_vector,
            cross_product_3d(omega_vector, scratch_data.quadrature_points[q]));
        }

      for (unsigned int j = 0; j < n_dofs; ++j)
        {
          const auto &phi_u_j = scratch_data.phi_u[q][j];
          if (dim == 2)
            strong_jacobian[q][j] +=
              2 * omega_z * (-1.) * cross_product_2d(phi_u_j);
          else if (dim == 3)
            strong_jacobian[q][j] +=
              2 * cross_product_3d(omega_vector, phi_u_j);
        }


      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const auto phi_u_i = scratch_data.phi_u[q][i];
          for (unsigned int j = 0; j < n_dofs; ++j)
            {
              const auto &phi_u_j = scratch_data.phi_u[q][j];

              if (dim == 2)
                local_matrix(i, j) += 2 * omega_z * (-1.) *
                                      cross_product_2d(phi_u_j) * phi_u_i *
                                      JxW[q];

              else if (dim == 3)
                local_matrix(i, j) += 2 *
                                      cross_product_3d(omega_vector, phi_u_j) *
                                      phi_u_i * JxW[q];
            }
        }
    }
}

template <int dim>
void
GLSNavierStokesAssemblerSRF<dim>::assemble_rhs(
  NavierStokesScratchData<dim> &        scratch_data,
  StabilizedMethodsTensorCopyData<dim> &copy_data)
{
  // Loop and quadrature informations
  const auto &       JxW               = scratch_data.JxW;
  const auto &       quadrature_points = scratch_data.quadrature_points;
  const unsigned int n_q_points        = scratch_data.n_q_points;
  const unsigned int n_dofs            = scratch_data.n_dofs;

  // Copy data elements
  auto &strong_residual = copy_data.strong_residual;
  auto &local_rhs       = copy_data.local_rhs;

  // SRF Source term
  //----------------------------------
  // Angular velocity of the rotating frame. This is always a 3D vector even
  // in 2D.
  Tensor<1, dim> omega_vector;

  double omega_z  = velocity_sources.omega_z;
  omega_vector[0] = velocity_sources.omega_x;
  omega_vector[1] = velocity_sources.omega_y;
  if (dim == 3)
    omega_vector[2] = velocity_sources.omega_z;


  // Loop over the quadrature points
  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      // Velocity
      const std::vector<Tensor<1, dim>> velocity = {
        scratch_data.velocity_values[q]};

      if (dim == 2)
        {
          strong_residual[q] +=
            2 * omega_z * (-1.) * cross_product_2d(velocity[0]);
          auto centrifugal =
            omega_z * (-1.) *
            cross_product_2d(
              omega_z * (-1.) *
              cross_product_2d(scratch_data.quadrature_points[q]));
          strong_residual[q] += centrifugal;
        }
      else // dim == 3
        {
          strong_residual[q] += 2 * cross_product_3d(omega_vector, velocity[0]);
          strong_residual[q] += cross_product_3d(
            omega_vector,
            cross_product_3d(omega_vector, scratch_data.quadrature_points[q]));
        }

      for (unsigned int i = 0; i < n_dofs; ++i)
        {
          const auto phi_u_i = scratch_data.phi_u[q][i];

          if (dim == 2)
            {
              local_rhs(i) += -2 * omega_z * (-1.) *
                              cross_product_2d(velocity[0]) * phi_u_i * JxW[q];
              auto centrifugal =
                omega_z * (-1.) *
                cross_product_2d(omega_z * (-1.) *
                                 cross_product_2d(quadrature_points[q]));
              local_rhs(i) += -centrifugal * phi_u_i * JxW[q];
            }
          else if (dim == 3)
            {
              local_rhs(i) += -2 * cross_product_3d(omega_vector, velocity[0]) *
                              phi_u_i * JxW[q];
              local_rhs(i) +=
                -cross_product_3d(omega_vector,
                                  cross_product_3d(omega_vector,
                                                   quadrature_points[q])) *
                phi_u_i * JxW[q];
            }
        }
    }
}



template class GLSNavierStokesAssemblerSRF<2>;
template class GLSNavierStokesAssemblerSRF<3>;
