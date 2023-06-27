/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------*/


#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Define all the parameters that can be specified in the .prm file
struct Settings
{
  bool
  try_parse(const std::string &prm_filename);

  enum PreconditionerType
  {
    amg,
    gmg,
    ilu,
    none
  };

  enum GeometryType
  {
    hypercube
  };

  enum SourceTermType
  {
    zero,
    mms
  };

  PreconditionerType preconditioner;
  GeometryType       geometry;
  SourceTermType     source_term;

  int          dimension;
  unsigned int u_order;
  unsigned int p_order;
  unsigned int number_of_cycles;
  unsigned int initial_refinement;
  unsigned int repetitions;
  bool         nonlinearity;
  bool         output;
  std::string  output_name;
  std::string  output_path;
};

bool
Settings::try_parse(const std::string &prm_filename)
{
  ParameterHandler prm;
  prm.declare_entry("dim",
                    "2",
                    Patterns::Integer(),
                    "The problem dimension <2|3>");
  prm.declare_entry("u order",
                    "1",
                    Patterns::Integer(),
                    "Order of FE element for the u <1|2|3>");
  prm.declare_entry("p order",
                    "1",
                    Patterns::Integer(),
                    "Order of FE element for the p <1|2|3>");
  prm.declare_entry("number of cycles",
                    "1",
                    Patterns::Integer(),
                    "Number of cycles <1 up to 9-dim >");
  prm.declare_entry("geometry",
                    "hypercube",
                    Patterns::Selection("hypercube"),
                    "Geometry <hypercube>");
  prm.declare_entry("initial refinement",
                    "1",
                    Patterns::Integer(),
                    "Global refinement 1st cycle");
  prm.declare_entry("repetitions",
                    "1",
                    Patterns::Integer(),
                    "Repetitions in one direction for the hyperrectangle");
  prm.declare_entry("nonlinearity",
                    "false",
                    Patterns::Bool(),
                    "Add exponential nonlinearity <true|false>");
  prm.declare_entry("output",
                    "true",
                    Patterns::Bool(),
                    "Output vtu files <true|false>");
  prm.declare_entry("output name",
                    "solution",
                    Patterns::FileName(),
                    "Name for vtu files");
  prm.declare_entry("output path",
                    "./",
                    Patterns::FileName(),
                    "Path for vtu output files");
  prm.declare_entry("preconditioner",
                    "AMG",
                    Patterns::Selection("AMG|GMG|ILU|none"),
                    "GMRES Preconditioner <AMG|GMG|ILU|none>");
  prm.declare_entry("source term",
                    "zero",
                    Patterns::Selection("zero|mms"),
                    "Source term <zero|mms>");

  if (prm_filename.size() == 0)
    {
      std::cout
        << "****  Error: No input file provided!\n"
        << "****  Error: Call this program as './matrix_based_non_linear_poisson input.prm\n"
        << '\n'
        << "****  You may want to use one of the input files in this\n"
        << "****  directory, or use the following default values\n"
        << "****  to create an input file:\n";
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        prm.print_parameters(std::cout, ParameterHandler::Text);
      return false;
    }

  try
    {
      prm.parse_input(prm_filename);
    }
  catch (std::exception &e)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cerr << e.what() << std::endl;
      return false;
    }

  if (prm.get("preconditioner") == "AMG")
    this->preconditioner = amg;
  else if (prm.get("preconditioner") == "GMG")
    this->preconditioner = gmg;
  else if (prm.get("preconditioner") == "ILU")
    this->preconditioner = ilu;
  else if (prm.get("preconditioner") == "none")
    this->preconditioner = none;
  else
    AssertThrow(false, ExcNotImplemented());

  if (prm.get("geometry") == "hypercube")
    this->geometry = hypercube;
  else
    AssertThrow(false, ExcNotImplemented());

  if (prm.get("source term") == "zero")
    this->source_term = zero;
  else if (prm.get("source term") == "mms")
    this->source_term = mms;
  else
    AssertThrow(false, ExcNotImplemented());

  this->dimension          = prm.get_integer("dim");
  this->u_order            = prm.get_integer("u order");
  this->p_order            = prm.get_integer("p order");
  this->number_of_cycles   = prm.get_integer("number of cycles");
  this->initial_refinement = prm.get_integer("initial refinement");
  this->repetitions        = prm.get_integer("repetitions");
  this->nonlinearity       = prm.get_bool("nonlinearity");
  this->output             = prm.get_bool("output");
  this->output_name        = prm.get("output name");
  this->output_path        = prm.get("output path");

  return true;
}

template <int dim>
class AnalyticalSolution : public Function<dim>
{
public:
  AnalyticalSolution()
    : Function<dim>(dim + 1)
  {}

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &value) const override;
};

template <int dim>
void
AnalyticalSolution<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &  values) const
{
  AssertDimension(values.size(), dim + 1);

  values(0) = std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  values(1) = std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  values(2) = std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
}

template <int dim>
class FirstSourceTerm : public Function<dim>
{
public:
  FirstSourceTerm()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  template <typename number>
  number
  value(const Point<dim, number> &p, const unsigned int component = 0) const;
};

template <int dim>
template <typename number>
number
FirstSourceTerm<dim>::value(const Point<dim, number> &p,
                            const unsigned int        component) const
{
  number result;
  if (component == 0)
    result = (-2.0 * Utilities::fixed_power<2, double>(numbers::PI) - 1) *
             std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  else if (component == 1)
    result = (-2.0 * Utilities::fixed_power<2, double>(numbers::PI) - 1) *
             std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  else
    result = 0.0;
  return result;
}

template <int dim>
double
FirstSourceTerm<dim>::value(const Point<dim> & p,
                            const unsigned int component) const
{
  return value<double>(p, component);
}

template <int dim>
class SecondSourceTerm : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p,
        const unsigned int /* component */ = 0) const override
  {
    return -numbers::PI * (2.0 * numbers::PI * std::sin(numbers::PI * p[0]) *
                             std::sin(numbers::PI * p[1]) +
                           std::sin(numbers::PI * (p[0] + p[1])));
  }
};

template <int dim>
class BoundaryFunction : public Function<dim>
{
public:
  virtual double
  value(const Point<dim> &p,
        const unsigned int /* component */ = 0) const override
  {
    return p[0];
  }
};

// Matrix-free helper function
template <int dim, typename Number>
VectorizedArray<Number>
evaluate_function(const Function<dim> &                      function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
  VectorizedArray<Number> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      result[v] = function.value(p);
    }
  return result;
}

// Matrix-free helper function
template <int dim, typename Number, int components>
Tensor<1, dim, VectorizedArray<Number>>
evaluate_function(const Function<dim> &                      function,
                  const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
  Tensor<1, dim, VectorizedArray<Number>> result;
  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
    {
      Point<dim> p;
      for (unsigned int d = 0; d < dim; ++d)
        p[d] = p_vectorized[d][v];
      for (unsigned int d = 0; d < components; ++d)
        result[d][v] = function.value(p, d);
    }
  return result;
}

// Matrix-free differential operator for a vector-valued problem
template <int dim, typename number>
class VectorValuedOperator
  : public MatrixFreeOperators::Base<dim,
                                     LinearAlgebra::distributed::Vector<number>>
{
public:
  using value_type = number;

  using UIntegrator = FEEvaluation<dim, -1, 0, dim + 1, number>;

  VectorValuedOperator();

  virtual void
  clear() override;

  void
  reinit_operator_parameters(const Settings &parameters);

  void
  evaluate_newton_step(
    const LinearAlgebra::distributed::Vector<number> &newton_step);

  virtual void
  compute_diagonal() override;

  const TrilinosWrappers::SparseMatrix &
  get_system_matrix(const AffineConstraints<number> &constraints);

private:
  virtual void
  apply_add(
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src) const override;

  void
  local_apply(const MatrixFree<dim, number> &                   data,
              LinearAlgebra::distributed::Vector<number> &      dst,
              const LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  local_compute_diagonal(UIntegrator &u_phi) const;

  mutable TrilinosWrappers::SparseMatrix system_matrix;
  Settings                               parameters;
  Table<2, VectorizedArray<number>>      nonlinear_values;
};

template <int dim, typename number>
VectorValuedOperator<dim, number>::VectorValuedOperator()
  : MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
{
  system_matrix.clear();
}

template <int dim, typename number>
void
VectorValuedOperator<dim, number>::clear()
{
  nonlinear_values.reinit(0, 0);
  MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>::
    clear();
}

template <int dim, typename number>
void
VectorValuedOperator<dim, number>::reinit_operator_parameters(
  const Settings &parameters)
{
  this->parameters = parameters;
}

template <int dim, typename number>
void
VectorValuedOperator<dim, number>::evaluate_newton_step(
  const LinearAlgebra::distributed::Vector<number> &newton_step)
{
  const unsigned int n_cells = this->data->n_cell_batches();
  UIntegrator        phi(*this->data);

  nonlinear_values.reinit(n_cells, phi.n_q_points);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(newton_step);
      phi.evaluate(EvaluationFlags::values);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          if (parameters.nonlinearity)
            nonlinear_values(cell, q) = VectorizedArray<number>(0.0);
          else
            nonlinear_values(cell, q) = VectorizedArray<number>(0.0);
        }
    }
}

template <int dim, typename number>
void
VectorValuedOperator<dim, number>::local_apply(
  const MatrixFree<dim, number> &                   data,
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  UIntegrator u_phi(data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      AssertDimension(nonlinear_values.size(0),
                      u_phi.get_matrix_free().n_cell_batches());
      AssertDimension(nonlinear_values.size(1), u_phi.n_q_points);

      u_phi.reinit(cell);
      u_phi.gather_evaluate(src,
                            EvaluationFlags::values |
                              EvaluationFlags::gradients);

      for (unsigned int q = 0; q < u_phi.n_q_points; ++q)
        {
          // First term
          u_phi.submit_gradient(-u_phi.get_gradient(q), q);

          // // Second term
          // p_phi.submit_value(-u_phi.get_divergence(q), q);

          // Third term
          // VectorizedArray<double>                 p_value =
          // p_phi.get_value(q); Tensor<1, dim, VectorizedArray<double>>
          // identity_by_p; for (unsigned int d = 0; d < dim; ++d)
          //   identity_by_p[d] = p_value;

          // u_phi.submit_value(-identity_by_p, q);

          // // Fourth term
          // p_phi.submit_gradient(-p_phi.get_gradient(q), q);
        }

      u_phi.integrate_scatter(EvaluationFlags::values |
                                EvaluationFlags::gradients,
                              dst);
    }
}

template <int dim, typename number>
void
VectorValuedOperator<dim, number>::apply_add(
  LinearAlgebra::distributed::Vector<number> &      dst,
  const LinearAlgebra::distributed::Vector<number> &src) const
{
  this->data->cell_loop(&VectorValuedOperator::local_apply, this, dst, src);
}

template <int dim, typename number>
void
VectorValuedOperator<dim, number>::local_compute_diagonal(
  UIntegrator &u_phi) const
{
  AssertDimension(nonlinear_values.size(0),
                  u_phi.get_matrix_free().n_cell_batches());
  AssertDimension(nonlinear_values.size(1), u_phi.n_q_points);

  // u_phi.reinit(cell);
  u_phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

  for (unsigned int q = 0; q < u_phi.n_q_points; ++q)
    {
      // First term
      u_phi.submit_gradient(-u_phi.get_gradient(q), q);

      // // Second term
      // p_phi.submit_value(-u_phi.get_divergence(q), q);

      // Third term
      // VectorizedArray<double>                 p_value =
      // p_phi.get_value(q); Tensor<1, dim, VectorizedArray<double>>
      // identity_by_p; for (unsigned int d = 0; d < dim; ++d)
      //   identity_by_p[d] = p_value;

      // u_phi.submit_value(-identity_by_p, q);

      // // Fourth term
      // p_phi.submit_gradient(-p_phi.get_gradient(q), q);
    }

  u_phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
}

template <int dim, typename number>
void
VectorValuedOperator<dim, number>::compute_diagonal()
{
  this->inverse_diagonal_entries.reset(
    new DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>());
  LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
    this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(inverse_diagonal);

  MatrixFreeTools::compute_diagonal(*this->data,
                                    inverse_diagonal,
                                    &VectorValuedOperator::local_compute_diagonal,
                                    this);

  for (auto &diagonal_element : inverse_diagonal)
    {
      diagonal_element =
        (std::abs(diagonal_element) > 1.0e-10) ? (1.0 / diagonal_element) : 1.0;
    }
}

template <int dim, typename number>
const TrilinosWrappers::SparseMatrix &
VectorValuedOperator<dim, number>::get_system_matrix(
  const AffineConstraints<number> &constraints)
{
  if (system_matrix.m() == 0 && system_matrix.n() == 0)
    {
      AffineConstraints<number> constraints_copy;
      constraints_copy.copy_from(constraints);

      const auto &dof_handler = this->get_matrix_free()->get_dof_handler();
      const auto &mg_level    = this->get_matrix_free()->get_mg_level();

      TrilinosWrappers::SparsityPattern dsp(
        mg_level != numbers::invalid_unsigned_int ?
          dof_handler.locally_owned_mg_dofs(mg_level) :
          dof_handler.locally_owned_dofs(),
        dof_handler.get_triangulation().get_communicator());

      if (mg_level != numbers::invalid_unsigned_int)
        MGTools::make_sparsity_pattern(dof_handler, dsp, mg_level, constraints);
      else
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);


      dsp.compress();
      system_matrix.reinit(dsp);

      MatrixFreeTools::compute_matrix(
        *this->data,
        constraints,
        system_matrix,
        &VectorValuedOperator::local_compute_diagonal,
        this);
    }
  return system_matrix;
}

// Main class for a dummy vector-valued problem given by
// ∇^2 u = p + f_1 and ∇^2 p = ∇ · u + f_2 using Newton's method
// and the matrix-free approach.
template <int dim>
class VectorValuedProblem
{
public:
  VectorValuedProblem(const Settings &parameters);

  void
  run();

private:
  void
  make_grid();

  void
  setup_system();

  void
  setup_gmg();

  void
  assemble_matrix();

  void
  assemble_coarse_matrix();

  void
  evaluate_residual(
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src) const;

  void
  local_evaluate_residual(
    const MatrixFree<dim, double> &                   data,
    LinearAlgebra::distributed::Vector<double> &      dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &     cell_range) const;

  void
  assemble_rhs();

  double
  compute_residual(const double alpha);

  void
  compute_update();

  void
  solve();

  void
  compute_solution_norm() const;

  void
  compute_l2_error() const;

  void
  output_results(const unsigned int cycle) const;


  parallel::distributed::Triangulation<dim> triangulation;

  Settings parameters;

  const unsigned int u_order;
  const unsigned int p_order;

  const MappingQ<dim> mapping;

  FESystem<dim>   fe_system;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;
  using SystemMatrixType = VectorValuedOperator<dim, double>;
  SystemMatrixType  system_matrix;
  MGConstrainedDoFs mg_constrained_dofs;
  using LevelMatrixType = VectorValuedOperator<dim, double>;
  MGLevelObject<LevelMatrixType>                            mg_matrices;
  MGLevelObject<LevelMatrixType>                            mg_interface_in;
  MGLevelObject<LevelMatrixType>                            mg_interface_out;
  MGLevelObject<LinearAlgebra::distributed::Vector<double>> mg_solution;
  MGTransferMatrixFree<dim, double>                         mg_transfer;
  MGLevelObject<AffineConstraints<double>>                  level_constraints;
  TrilinosWrappers::SparseMatrix                            mb_system_matrix;
  TrilinosWrappers::SparseMatrix mb_system_matrix_coarse;

  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> newton_update;
  LinearAlgebra::distributed::Vector<double> system_rhs;

  unsigned int       linear_iterations;
  ConditionalOStream pcout;
  TimerOutput        computing_timer;
};

template <int dim>
VectorValuedProblem<dim>::VectorValuedProblem(const Settings &parameters)
  : triangulation(
      MPI_COMM_WORLD,
      Triangulation<dim>::limit_level_difference_at_vertices,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
  , parameters(parameters)
  , u_order(parameters.u_order)
  , p_order(parameters.p_order)
  , mapping(parameters.u_order)
  , fe_system(FE_Q<dim>(parameters.u_order), dim + 1)
  , dof_handler(triangulation)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  , computing_timer(MPI_COMM_WORLD,
                    pcout,
                    TimerOutput::never,
                    TimerOutput::wall_times)
{}

template <int dim>
void
VectorValuedProblem<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "make grid");

  switch (parameters.geometry)
    {
        case Settings::hypercube: {
          GridGenerator::hyper_cube(triangulation, 0, 1.0, true);
          break;
        }
    }

  triangulation.refine_global(parameters.initial_refinement);
}

template <int dim>
void
VectorValuedProblem<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup system");

  system_matrix.clear();

  system_matrix.reinit_operator_parameters(parameters);

  dof_handler.distribute_dofs(fe_system);

  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  // Set homogeneous constraints for the matrix-free operator
  // Create zero BCs for the delta.
  // Left wall
  VectorTools::interpolate_boundary_values(
    dof_handler, 0, Functions::ZeroFunction<dim>(dim + 1), constraints);

  VectorTools::interpolate_boundary_values(
    dof_handler, 1, Functions::ZeroFunction<dim>(dim + 1), constraints);

  VectorTools::interpolate_boundary_values(
    dof_handler, 2, Functions::ZeroFunction<dim>(dim + 1), constraints);

  VectorTools::interpolate_boundary_values(
    dof_handler, 3, Functions::ZeroFunction<dim>(dim + 1), constraints);

  constraints.close();

  {
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags =
      (update_values | update_gradients | update_JxW_values |
       update_quadrature_points);
    auto system_mf_storage = std::make_shared<MatrixFree<dim, double>>();
    system_mf_storage->reinit(mapping,
                              dof_handler,
                              constraints,
                              QGauss<1>(u_order + 1),
                              additional_data);

    system_matrix.initialize(system_mf_storage);
  }

  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(newton_update);
  system_matrix.initialize_dof_vector(system_rhs);

  MPI_Comm               mpi_communicator(MPI_COMM_WORLD);
  IndexSet               locally_owned_dofs = dof_handler.locally_owned_dofs();
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             mpi_communicator,
                                             locally_relevant_dofs);
  constraints.condense(dsp);

  mb_system_matrix.reinit(locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);
}

template <int dim>
void
VectorValuedProblem<dim>::setup_gmg()
{
  TimerOutput::Scope t(computing_timer, "setup GMG");

  dof_handler.distribute_mg_dofs();

  mg_matrices.clear_elements();
  mg_interface_in.clear_elements();
  mg_interface_out.clear_elements();

  const unsigned int nlevels = triangulation.n_global_levels();
  mg_matrices.resize(0, nlevels - 1);
  mg_interface_in.resize(0, nlevels - 1);
  mg_interface_out.resize(0, nlevels - 1);
  mg_solution.resize(0, nlevels - 1);
  level_constraints.resize(0, nlevels - 1);

  const std::set<types::boundary_id> dirichlet_boundary_ids = {0, 1, 2, 3};
  mg_constrained_dofs.initialize(dof_handler);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                     dirichlet_boundary_ids);

  mg_transfer.initialize_constraints(mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  for (unsigned int level = 0; level < nlevels; ++level)
    {
      mg_matrices[level].reinit_operator_parameters(parameters);
      const IndexSet relevant_dofs =
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);

      // AffineConstraints<double> level_constraints;
      level_constraints[level].reinit(relevant_dofs);
      level_constraints[level].add_lines(
        mg_constrained_dofs.get_boundary_indices(level));
      level_constraints[level].close();

      typename MatrixFree<dim, double>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::partition_color;
      additional_data.mapping_update_flags =
        (update_values | update_gradients | update_JxW_values |
         update_quadrature_points);
      additional_data.mg_level = level;
      auto mg_mf_storage_level = std::make_shared<MatrixFree<dim, double>>();
      mg_mf_storage_level->reinit(mapping,
                                  dof_handler,
                                  level_constraints[level],
                                  QGauss<1>(u_order + 1),
                                  additional_data);

      mg_matrices[level].initialize(mg_mf_storage_level,
                                    mg_constrained_dofs,
                                    level);
      mg_matrices[level].initialize_dof_vector(mg_solution[level]);

      mg_interface_in[level].initialize(mg_mf_storage_level,
                                        mg_constrained_dofs,
                                        level);
      mg_interface_out[level].initialize(mg_mf_storage_level,
                                         mg_constrained_dofs,
                                         level);
    }
}

template <int dim>
void
VectorValuedProblem<dim>::assemble_matrix()
{
  const QGauss<dim> quadrature_formula(u_order + 1);

  mb_system_matrix = 0;

  FEValues<dim> fe_values(fe_system,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
  const unsigned int n_q_points    = fe_values.n_quadrature_points;

  // const FEValuesExtractors::Vector u(0);
  // const FEValuesExtractors::Scalar p(dim);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_matrix = 0.0;

          fe_values.reinit(cell);

          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              const double dx = fe_values.JxW(q);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // const Tensor<1, dim> phi_i      = fe_values[u].value(i, q);
                  // const Tensor<2, dim> grad_phi_i = fe_values.shape_grad(i,
                  // q)[fe_system.system_to_component_index(i).first]; const
                  // double         psi_i      = fe_values[p].value(i, q); const
                  // Tensor<1, dim> grad_psi_i = fe_values[p].gradient(i, q);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      // const Tensor<1, dim> phi_j = fe_values[u].value(j, q);
                      // const Tensor<2, dim> grad_phi_j =
                      //   fe_values.shape_grad(j, q);
                      // const double div_phi_j = fe_values[u].divergence(j, q);
                      // const double psi_j     = fe_values[p].value(j, q);
                      // const Tensor<1, dim> grad_psi_j =
                      //   fe_values[p].gradient(j, q);
                      // Tensor<1, dim> identity;
                      // identity[0] = 1;
                      // identity[1] = 1;

                      cell_matrix(i, j) -= 0;
                      // (scalar_product(grad_phi_i, grad_phi_j) /* +
                      //  (psi_i * div_phi_j) + (phi_i * psi_j * identity) +
                      //  (grad_psi_i * grad_psi_j) */) *
                      // dx;
                    }
                }
            }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 mb_system_matrix);
        }
    }

  mb_system_matrix.compress(VectorOperation::add);
}

template <int dim>
void
VectorValuedProblem<dim>::assemble_coarse_matrix()
{
  QGauss<dim> quadrature_formula(u_order + 1);

  FEValues<dim> fe_values(fe_system,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
  const unsigned int n_q_points    = fe_values.n_quadrature_points;

  const FEValuesExtractors::Vector u(0);
  const FEValuesExtractors::Scalar p(dim);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<AffineConstraints<double>> boundary_constraints(
    triangulation.n_global_levels());
  const IndexSet dof_set =
    DoFTools::extract_locally_relevant_level_dofs(dof_handler, 0);
  boundary_constraints[0].reinit(dof_set);
  boundary_constraints[0].add_lines(
    mg_constrained_dofs.get_refinement_edge_indices(0));
  boundary_constraints[0].add_lines(
    mg_constrained_dofs.get_boundary_indices(0));
  boundary_constraints[0].close();

  for (const auto &cell : dof_handler.cell_iterators())
    if (cell->level_subdomain_id() == triangulation.locally_owned_subdomain())
      {
        cell_matrix = 0;

        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double dx = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const Tensor<1, dim> phi_i      = fe_values[u].value(i, q);
                const Tensor<2, dim> grad_phi_i = fe_values[u].gradient(i, q);
                const double         psi_i      = fe_values[p].value(i, q);
                const Tensor<1, dim> grad_psi_i = fe_values[p].gradient(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    // const Tensor<1, dim> phi_j = fe_values[u].value(j, q);
                    const Tensor<2, dim> grad_phi_j =
                      fe_values[u].gradient(j, q);
                    const double div_phi_j = fe_values[u].divergence(j, q);
                    const double psi_j     = fe_values[p].value(j, q);
                    const Tensor<1, dim> grad_psi_j =
                      fe_values[p].gradient(j, q);
                    Tensor<1, dim> identity;
                    identity[0] = 1;
                    identity[1] = 0;

                    cell_matrix(i, j) -=
                      (scalar_product(grad_phi_i, grad_phi_j) +
                       (psi_i * div_phi_j) + (phi_i * psi_j * identity) +
                       (grad_psi_i * grad_psi_j)) *
                      dx;
                  }
              }
          }

        cell->get_mg_dof_indices(local_dof_indices);

        boundary_constraints[0].distribute_local_to_global(
          cell_matrix, local_dof_indices, mb_system_matrix_coarse);
      }

  mb_system_matrix_coarse.compress(VectorOperation::add);
}

template <int dim>
void
VectorValuedProblem<dim>::evaluate_residual(
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src) const
{
  auto matrix_free = system_matrix.get_matrix_free();

  matrix_free->cell_loop(
    &VectorValuedProblem::local_evaluate_residual, this, dst, src, true);
}

template <int dim>
void
VectorValuedProblem<dim>::local_evaluate_residual(
  const MatrixFree<dim, double> &                   data,
  LinearAlgebra::distributed::Vector<double> &      dst,
  const LinearAlgebra::distributed::Vector<double> &src,
  const std::pair<unsigned int, unsigned int> &     cell_range) const
{
  FEEvaluation<dim, -1, 0, dim, double> phi(data);

  FirstSourceTerm<dim>  first_source_term_function;
  SecondSourceTerm<dim> second_source_term_function;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values_plain(src);
      phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          VectorizedArray<double> second_source_value =
            VectorizedArray<double>(0.0);
          Tensor<1, dim, VectorizedArray<double>> first_source_value;
          Point<dim, VectorizedArray<double>>     point_batch =
            phi.quadrature_point(q);
          if (parameters.source_term == Settings::mms)
            {
              first_source_value =
                evaluate_function<dim, double, dim>(first_source_term_function,
                                                    point_batch);

              second_source_value =
                evaluate_function<dim, double>(second_source_term_function,
                                               point_batch);
            }

          // // First term + fourth term
          phi.submit_gradient(-phi.get_gradient(q), q);

          // Second term  + second source term
          // Tensor<1,dim> source_terms;
          // source_terms[0] = first_source_value;
          // source_terms[1] = second_source_value;
          // phi.submit_value(-source_terms,q);

          // p_phi.submit_value(-u_phi.get_divergence(q) - second_source_value,
          // q);

          // // Third term + first source term
          // VectorizedArray<double>                 p_value =
          // p_phi.get_value(q); Tensor<1, dim, VectorizedArray<double>>
          // identity_by_p; for (unsigned int d = 0; d < dim; ++d)
          //   identity_by_p[d] = p_value;

          // u_phi.submit_value(-identity_by_p - first_source_value, q);

          // // Fourth term
          // p_phi.submit_gradient(-p_phi.get_gradient(q), q);
        }

      phi.integrate_scatter(EvaluationFlags::values |
                              EvaluationFlags::gradients,
                            dst);
    }
}

template <int dim>
void
VectorValuedProblem<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "assemble right hand side");

  evaluate_residual(system_rhs, solution);

  system_rhs *= -1.0;
}

template <int dim>
double
VectorValuedProblem<dim>::compute_residual(const double alpha)
{
  TimerOutput::Scope t(computing_timer, "compute residual");

  LinearAlgebra::distributed::Vector<double> residual;
  LinearAlgebra::distributed::Vector<double> evaluation_point;

  system_matrix.initialize_dof_vector(residual);
  system_matrix.initialize_dof_vector(evaluation_point);

  evaluation_point = solution;
  if (alpha > 1e-12)
    {
      evaluation_point.add(alpha, newton_update);
    }

  evaluate_residual(residual, evaluation_point);

  return residual.l2_norm();
}

template <int dim>
void
VectorValuedProblem<dim>::compute_update()
{
  TimerOutput::Scope t(computing_timer, "compute update");

  solution.update_ghost_values();

  SolverControl solver_control(200, 1.e-8 * system_rhs.l2_norm(), true, true);
  SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);

  newton_update = 0.0;

  switch (parameters.preconditioner)
    {
        case Settings::amg: {
          TrilinosWrappers::PreconditionAMG                 preconditioner;
          TrilinosWrappers::PreconditionAMG::AdditionalData data;

          if (parameters.u_order > 1)
            data.higher_order_elements = true;

          data.elliptic      = false;
          data.smoother_type = "Jacobi";

          system_matrix.evaluate_newton_step(solution);
          // assemble_matrix();
          preconditioner.initialize(
            system_matrix.get_system_matrix(constraints), data);
          gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
          break;
        }
        case Settings::gmg: {
          system_matrix.evaluate_newton_step(solution);

          mg_transfer.interpolate_to_mg(dof_handler, mg_solution, solution);

          // Set up smoother: Jacobi Smoother
          using SmootherTypeJacobi = PreconditionJacobi<LevelMatrixType>;
          MGSmootherPrecondition<LevelMatrixType,
                                 SmootherTypeJacobi,
                                 LinearAlgebra::distributed::Vector<double>>
            mg_smoother_jacobi;
          for (unsigned int level = 0; level < triangulation.n_global_levels();
               ++level)
            {
              mg_matrices[level].evaluate_newton_step(mg_solution[level]);
              mg_matrices[level].compute_diagonal();
            }
          mg_smoother_jacobi.initialize(
            mg_matrices,
            typename SmootherTypeJacobi::AdditionalData(
              u_order == 1 ? 0.6667 : (u_order == 2 ? 0.56 : 0.47)));
          mg_smoother_jacobi.set_steps(6);

          // Set up preconditioned coarse-grid solver
          SolverControl coarse_solver_control(200,
                                              1.e-8 * system_rhs.l2_norm(),
                                              true,
                                              true);

          SolverGMRES<LinearAlgebra::distributed::Vector<double>> coarse_solver(
            coarse_solver_control,
            SolverGMRES<LinearAlgebra::distributed::Vector<double>>::
              AdditionalData(50, true));

          // Coarse-grid solver optional AMG preconditioner
          // TrilinosWrappers::PreconditionAMG                 precondition_amg;
          // TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          // precondition_amg.initialize(
          //   mg_matrices[0].get_system_matrix(constraints), amg_data);

          PreconditionIdentity identity;
          MGCoarseGridIterativeSolver<
            LinearAlgebra::distributed::Vector<double>,
            SolverGMRES<LinearAlgebra::distributed::Vector<double>>,
            LevelMatrixType,
            PreconditionIdentity>
            mg_coarse(coarse_solver, mg_matrices[0], identity);

          // Set up multigrid
          mg::Matrix<LinearAlgebra::distributed::Vector<double>> mg_matrix(
            mg_matrices);

          // TODO correct mg interface matrices for adaptive meshes
          MGLevelObject<
            MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
            mg_interface_matrices_in;
          MGLevelObject<
            MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>>
            mg_interface_matrices_out;
          mg_interface_matrices_in.resize(0,
                                          triangulation.n_global_levels() - 1);
          mg_interface_matrices_out.resize(0,
                                           triangulation.n_global_levels() - 1);
          for (unsigned int level = 0; level < triangulation.n_global_levels();
               ++level)
            {
              mg_interface_matrices_in[level].initialize(mg_matrices[level]);
              mg_interface_matrices_out[level].initialize(mg_matrices[level]);
            }
          mg::Matrix<LinearAlgebra::distributed::Vector<double>>
            mg_interface_in(mg_interface_matrices_in);
          mg::Matrix<LinearAlgebra::distributed::Vector<double>>
            mg_interface_out(mg_interface_matrices_out);

          Multigrid<LinearAlgebra::distributed::Vector<double>> mg(
            mg_matrix,
            mg_coarse,
            mg_transfer,
            mg_smoother_jacobi,
            mg_smoother_jacobi);
          mg.set_edge_matrices(mg_interface_in, mg_interface_out);

          PreconditionMG<dim,
                         LinearAlgebra::distributed::Vector<double>,
                         MGTransferMatrixFree<dim, double>>
            preconditioner(dof_handler, mg, mg_transfer);

          gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
          break;
        }
        case Settings::ilu: {
          TrilinosWrappers::PreconditionILU                 preconditioner;
          TrilinosWrappers::PreconditionILU::AdditionalData data_ilu;
          system_matrix.evaluate_newton_step(solution);
          assemble_matrix();
          preconditioner.initialize(mb_system_matrix, data_ilu);

          gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
          break;
        }
        case Settings::none: {
          system_matrix.evaluate_newton_step(solution);
          PreconditionIdentity preconditioner;
          gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
          break;
        }
      default:
        Assert(
          false,
          ExcMessage(
            "This program supports only AMG, GMG and ILU as preconditioners for the GMRES solver."));
    }

  constraints.distribute(newton_update);

  linear_iterations = solver_control.last_step();

  solution.zero_out_ghost_values();
}


template <int dim>
void
VectorValuedProblem<dim>::solve()
{
  TimerOutput::Scope t(computing_timer, "solve");

  const unsigned int itmax = 14;
  const double       TOLf  = 1e-12;
  const double       TOLx  = 1e-10;

  Timer solver_timer;
  solver_timer.start();

  for (unsigned int newton_step = 1; newton_step <= itmax; ++newton_step)
    {
      assemble_rhs();
      compute_update();
      const double ERRx = newton_update.l2_norm();
      const double ERRf = compute_residual(1.0);
      solution.add(1.0, newton_update);

      pcout << "   Nstep " << newton_step << ", errf = " << ERRf
            << ", errx = " << ERRx << ", it = " << linear_iterations
            << std::endl;

      if (ERRf < TOLf || ERRx < TOLx)
        {
          solver_timer.stop();

          pcout << "Convergence step " << newton_step << " value " << ERRf
                << " (used wall time: " << solver_timer.wall_time() << " s)"
                << std::endl;

          break;
        }
      else if (newton_step == itmax)
        {
          solver_timer.stop();
          pcout << "WARNING: No convergence of Newton's method after "
                << newton_step << " steps." << std::endl;

          break;
        }
    }
}

template <int dim>
void
VectorValuedProblem<dim>::compute_solution_norm() const
{
  solution.update_ghost_values();

  const ComponentSelectFunction<dim> p_mask(dim, dim + 1);
  const ComponentSelectFunction<dim> u_mask(std::make_pair(0, dim), dim + 1);

  Vector<float> norm_per_cell(triangulation.n_active_cells());

  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    Functions::ZeroFunction<dim>(dim + 1),
                                    norm_per_cell,
                                    QGauss<dim>(u_order + 1),
                                    VectorTools::H1_seminorm,
                                    &u_mask);

  solution.zero_out_ghost_values();

  double u_h1_norm =
    VectorTools::compute_global_error(triangulation,
                                      norm_per_cell,
                                      VectorTools::H1_seminorm);

  solution.update_ghost_values();

  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    Functions::ZeroFunction<dim>(dim + 1),
                                    norm_per_cell,
                                    QGauss<dim>(p_order + 1),
                                    VectorTools::H1_seminorm,
                                    &p_mask);

  solution.zero_out_ghost_values();

  double p_h1_norm =
    VectorTools::compute_global_error(triangulation,
                                      norm_per_cell,
                                      VectorTools::H1_seminorm);

  pcout << "  u H1 seminorm: " << u_h1_norm << std::endl;
  pcout << "  p H1 seminorm: " << p_h1_norm << std::endl;
  pcout << std::endl;
}

template <int dim>
void
VectorValuedProblem<dim>::compute_l2_error() const
{
  solution.update_ghost_values();

  const ComponentSelectFunction<dim> p_mask(dim, dim + 1);
  const ComponentSelectFunction<dim> u_mask(std::make_pair(0, dim), dim + 1);

  Vector<float> error_per_cell(triangulation.n_active_cells());

  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    AnalyticalSolution<dim>(),
                                    error_per_cell,
                                    QGauss<dim>(u_order + 1),
                                    VectorTools::L2_norm,
                                    &u_mask);

  solution.zero_out_ghost_values();

  double u_l2_error = VectorTools::compute_global_error(triangulation,
                                                        error_per_cell,
                                                        VectorTools::L2_norm);

  solution.update_ghost_values();

  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    AnalyticalSolution<dim>(),
                                    error_per_cell,
                                    QGauss<dim>(p_order + 1),
                                    VectorTools::L2_norm,
                                    &p_mask);

  solution.zero_out_ghost_values();

  double p_l2_error = VectorTools::compute_global_error(triangulation,
                                                        error_per_cell,
                                                        VectorTools::L2_norm);

  pcout << "  u L2 norm: " << u_l2_error << std::endl;
  pcout << "  p L2 norm: " << p_l2_error << std::endl;
}

template <int dim>
void
VectorValuedProblem<dim>::output_results(const unsigned int cycle) const
{
  if (triangulation.n_global_active_cells() > 1e6)
    return;

  solution.update_ghost_values();

  std::vector<std::string> solution_names(dim, "u");
  solution_names.push_back("p");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    dealii_data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  dealii_data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           dealii_data_component_interpretation);

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    {
      subdomain(i) = triangulation.locally_owned_subdomain();
    }
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  DataOutBase::VtkFlags flags;
  flags.compression_level = DataOutBase::VtkFlags::best_speed;
  data_out.set_flags(flags);

  data_out.write_vtu_with_pvtu_record(parameters.output_path,
                                      parameters.output_name +
                                        std::to_string(dim),
                                      cycle,
                                      MPI_COMM_WORLD,
                                      3);

  solution.zero_out_ghost_values();
}

template <int dim>
void
VectorValuedProblem<dim>::run()
{
  {
    const unsigned int n_ranks =
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    const unsigned int n_vect_doubles = VectorizedArray<double>::size();
    const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

    std::string DAT_header = "START DATE: " + Utilities::System::get_date() +
                             ", TIME: " + Utilities::System::get_time() +
                             ", MATRIX-FREE SOLVER";
    std::string MPI_header = "Running with " + std::to_string(n_ranks) +
                             " MPI process" + (n_ranks > 1 ? "es" : "");
    std::string VEC_header =
      "Vectorization over " + std::to_string(n_vect_doubles) +
      " doubles = " + std::to_string(n_vect_bits) + " bits (" +
      Utilities::System::get_current_vectorization_level() +
      "), VECTORIZATION_LEVEL=" +
      std::to_string(DEAL_II_COMPILER_VECTORIZATION_LEVEL);
    std::string SOL_header = "Finite element space: " + fe_system.get_name();
    std::string PRECOND_header = "";
    if (parameters.preconditioner == Settings::amg)
      PRECOND_header = "Preconditioner: AMG";
    else if (parameters.preconditioner == Settings::gmg)
      PRECOND_header = "Preconditioner: GMG";
    else if (parameters.preconditioner == Settings::ilu)
      PRECOND_header = "Preconditioner: ILU";
    else if (parameters.preconditioner == Settings::none)
      PRECOND_header = "Preconditioner: none";
    std::string GEOMETRY_header = "";
    if (parameters.geometry == Settings::hypercube)
      GEOMETRY_header = "Geometry: hypercube";
    std::string SOURCE_header = "";
    if (parameters.source_term == Settings::zero)
      SOURCE_header = "Source term: zero";
    else if (parameters.source_term == Settings::mms)
      SOURCE_header = "Source term: according to MMS";
    std::string REFINE_header =
      "Initial refinement: " + std::to_string(parameters.initial_refinement);
    std::string CYCLES_header =
      "Total number of cycles: " + std::to_string(parameters.number_of_cycles);
    std::string NONLINEARITY_header = "";
    if (parameters.nonlinearity)
      NONLINEARITY_header = "Nonlinearity: true";
    else
      NONLINEARITY_header = "Nonlinearity: false";
    std::string REPETITIONS_header =
      "Repetitions in one direction: " + std::to_string(parameters.repetitions);

    pcout << std::string(80, '=') << std::endl;
    pcout << DAT_header << std::endl;
    pcout << std::string(80, '-') << std::endl;

    pcout << MPI_header << std::endl;
    pcout << VEC_header << std::endl;
    pcout << SOL_header << std::endl;
    pcout << PRECOND_header << std::endl;
    pcout << GEOMETRY_header << std::endl;
    pcout << SOURCE_header << std::endl;
    pcout << REFINE_header << std::endl;
    pcout << CYCLES_header << std::endl;
    pcout << NONLINEARITY_header << std::endl;
    pcout << REPETITIONS_header << std::endl;

    pcout << std::string(80, '=') << std::endl;
  }

  for (unsigned int cycle = 0; cycle < parameters.number_of_cycles; ++cycle)
    {
      pcout << std::string(80, '-') << std::endl;
      pcout << "Cycle " << cycle << std::endl;
      pcout << std::string(80, '-') << std::endl;

      if (cycle == 0)
        {
          make_grid();
        }
      else
        {
          triangulation.refine_global(1);
        }

      Timer timer;

      pcout << "Set up system..." << std::endl;
      setup_system();

      if (parameters.preconditioner == Settings::gmg)
        setup_gmg();

      pcout << "   Triangulation: " << triangulation.n_global_active_cells()
            << " cells" << std::endl;
      pcout << "   DoFHandler:    " << dof_handler.n_dofs() << " DoFs"
            << std::endl;

      if (parameters.preconditioner == Settings::gmg)
        for (unsigned int level = 0; level < triangulation.n_global_levels();
             ++level)
          pcout << "   MG Level " << level << ": " << dof_handler.n_dofs(level)
                << " DoFs, " << triangulation.n_cells(level) << " cells"
                << std::endl;

      pcout << std::endl;


      pcout << "Solve using Newton's method..." << std::endl;
      solve();
      pcout << std::endl;


      timer.stop();
      pcout << "Time for setup+solve (CPU/Wall) " << timer.cpu_time() << '/'
            << timer.wall_time() << " s" << std::endl;
      pcout << std::endl;

      if (parameters.output)
        {
          pcout << "Output results..." << std::endl;
          output_results(cycle);
        }

      compute_solution_norm();

      compute_l2_error();

      computing_timer.print_summary();
      computing_timer.reset();
    }
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize       mpi_init(argc, argv, 1);
  dealii::Utilities::System::MemoryStats stats;
  dealii::Utilities::System::get_memory_stats(stats);

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);

  Settings parameters;
  if (!parameters.try_parse((argc > 1) ? (argv[1]) : ""))
    return 0;

  try
    {
      switch (parameters.dimension)
        {
            case 2: {
              VectorValuedProblem<2> vector_valued_problem(parameters);
              vector_valued_problem.run();

              break;
            }

            case 3: {
              VectorValuedProblem<3> vector_valued_problem(parameters);
              vector_valued_problem.run();
              break;
            }

          default:
            Assert(
              false,
              ExcMessage(
                "This program only works in 2d and 3d and for element orders equal to 1, 2 or 3."));
        }

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
