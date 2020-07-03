/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */


// @sect3{Include files}

// The first few (many?) include files have already been used in the previous
// example, so we will not explain their meaning here again.
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;
enum simCase
{
  TaylorCouette,
  MMS
};

template <int dim>
struct ScratchData
{
  ScratchData(const Mapping<dim> &      mapping,
              const FiniteElement<dim> &fe,
              const unsigned int        quadrature_degree,
              const UpdateFlags         update_flags = update_values |
                                               update_gradients |
                                               update_quadrature_points |
                                               update_JxW_values,
              const UpdateFlags interface_update_flags =
                update_values | update_gradients | update_quadrature_points |
                update_JxW_values | update_normal_vectors)
    : fe_values(mapping, fe, QGauss<dim>(quadrature_degree), update_flags)
    , fe_interface_values(mapping,
                          fe,
                          QGauss<dim - 1>(quadrature_degree),
                          interface_update_flags)
  {}


  ScratchData(const ScratchData<dim> &scratch_data)
    : fe_values(scratch_data.fe_values.get_mapping(),
                scratch_data.fe_values.get_fe(),
                scratch_data.fe_values.get_quadrature(),
                scratch_data.fe_values.get_update_flags())
    , fe_interface_values(scratch_data.fe_values.get_mapping(),
                          scratch_data.fe_values.get_fe(),
                          scratch_data.fe_interface_values.get_quadrature(),
                          scratch_data.fe_interface_values.get_update_flags())
  {}

  FEValues<dim>          fe_values;
  FEInterfaceValues<dim> fe_interface_values;
};



struct CopyDataFace
{
  FullMatrix<double>                   cell_matrix;
  std::vector<types::global_dof_index> joint_dof_indices;
};



struct CopyData
{
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;
  std::vector<CopyDataFace>            face_data;

  template <class Iterator>
  void
  reinit(const Iterator &cell, unsigned int dofs_per_cell)
  {
    cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
    cell_rhs.reinit(dofs_per_cell);

    local_dof_indices.resize(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);
  }
};

template <int dim>
class RightHandSideMMS : public Function<dim>
{
public:
  RightHandSideMMS()
    : Function<dim>()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues()
    : Function<dim>()
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const;
};

template <int dim>
double
RightHandSideMMS<dim>::value(const Point<dim> &p,
                             const unsigned int /*component*/) const
{
  double return_value = 0.0;
  double x            = p(0);
  double y            = p(1);

  return_value = -2. * M_PI * M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);

  return return_value;
}

// As boundary values, we choose $x^2+y^2$ in 2D, and $x^2+y^2+z^2$ in 3D. This
// happens to be equal to the square of the vector from the origin to the
// point at which we would like to evaluate the function, irrespective of the
// dimension. So that is what we return:
template <int dim>
double
BoundaryValues<dim>::value(const Point<dim> &p,
                           const unsigned int /*component*/) const
{
  return p.square();
}

template <int dim>
class DGHeat
{
public:
  DGHeat(simCase scase, int refinementLevel);
  void
  run();
  double
  getL2Error()
  {
    return L2Error_;
  }

private:
  void
  make_grid();
  void
  make_cube_grid();
  void
  make_ring_grid();
  void
  setup_system();
  void
  assemble_system_DG();
  void
  assemble_system_CG();
  void
  solve();
  void
  output_results() const;
  void
  calculateL2Error();

  Triangulation<dim> triangulation;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  const MappingQ<dim> mapping;

  RightHandSideMMS<dim> right_hand_side;


  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
  Point<dim>     center;

  simCase simulationCase_;
  int     refinementLevel_;
  double  L2Error_;
};



template <int dim>
DGHeat<dim>::DGHeat(simCase scase, int refinementLevel)
  : fe(1)
  , mapping(1)
  , dof_handler(triangulation)
  , simulationCase_(scase)
  , refinementLevel_(refinementLevel)
{}

template <int dim>
void
DGHeat<dim>::make_grid()
{
  if (simulationCase_ == MMS)
    make_cube_grid();
  else if (simulationCase_ == TaylorCouette)
    make_ring_grid();
}

template <int dim>
void
DGHeat<dim>::make_cube_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(refinementLevel_);

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}

template <int dim>
void
DGHeat<dim>::make_ring_grid()
{
  const double inner_radius = 0.25, outer_radius = 1.0;
  if (dim == 2)
    center = Point<dim>(0, 0);
  GridGenerator::hyper_shell(
    triangulation, center, inner_radius, outer_radius, 10, true);

  triangulation.refine_global(refinementLevel_);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  std::cout << "Number of total cells: " << triangulation.n_cells()
            << std::endl;
}


template <int dim>
void
DGHeat<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


template <int dim>
void
DGHeat<dim>::assemble_system_DG()
{
  using Iterator = typename DoFHandler<dim>::active_cell_iterator;
  const BoundaryValues<dim> boundary_function;

  auto cell_worker = [&](const Iterator &  cell,
                         ScratchData<dim> &scratch_data,
                         CopyData &        copy_data) {
    const unsigned int n_dofs = scratch_data.fe_values.get_fe().dofs_per_cell;
    copy_data.reinit(cell, n_dofs);
    scratch_data.fe_values.reinit(cell);

    const auto &q_points = scratch_data.fe_values.get_quadrature_points();

    const FEValues<dim> &      fe_v = scratch_data.fe_values;
    const std::vector<double> &JxW  = fe_v.get_JxW_values();

    std::vector<double> f(q_points.size());
    right_hand_side.value_list(q_points, f);

    for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
      {
        auto beta_q = beta(q_points[point]);
        for (unsigned int i = 0; i < n_dofs; ++i)
          {
            for (unsigned int j = 0; j < n_dofs; ++j)
              {
                copy_data.cell_matrix(i, j) +=
                  fe_v.shape_grad(i, point)   // \nabla \phi_i
                  * fe_v.shape_grad(j, point) // \nabla \phi_j
                  * JxW[point];               // dx
              }
            // Right Hand Side
            copy_data.cell_rhs(i) +=
              (fe_v.shape_grad(i, point) * f[point] * JxW[point]);
          }
      }
  };

  auto boundary_worker = [&](const Iterator &    cell,
                             const unsigned int &face_no,
                             ScratchData<dim> &  scratch_data,
                             CopyData &          copy_data) {
    scratch_data.fe_interface_values.reinit(cell, face_no);
    const FEFaceValuesBase<dim> &fe_face =
      scratch_data.fe_interface_values.get_fe_face_values(0);

    const auto &q_points = fe_face.get_quadrature_points();

    const unsigned int n_facet_dofs        = fe_face.get_fe().n_dofs_per_cell();
    const std::vector<double> &        JxW = fe_face.get_JxW_values();
    const std::vector<Tensor<1, dim>> &normals = fe_face.get_normal_vectors();

    std::vector<double> g(q_points.size());
    boundary_function.value_list(q_points, g);

    for (unsigned int point = 0; point < q_points.size(); ++point)
      {
        //        const double beta_dot_n = beta(q_points[point]) *
        //        normals[point];

        //        if (beta_dot_n > 0)
        //          {
        //            for (unsigned int i = 0; i < n_facet_dofs; ++i)
        //              for (unsigned int j = 0; j < n_facet_dofs; ++j)
        //                copy_data.cell_matrix(i, j) +=
        //                  fe_face.shape_value(i, point)   // \phi_i
        //                  * fe_face.shape_value(j, point) // \phi_j
        //                  * beta_dot_n                    // \beta . n
        //                  * JxW[point];                   // dx
        //          }
        //        else
        //          for (unsigned int i = 0; i < n_facet_dofs; ++i)
        //            copy_data.cell_rhs(i) += -fe_face.shape_value(i, point) //
        //            \phi_i
        //                                     * g[point]                     //
        //                                     g
        //                                     * beta_dot_n                   //
        //                                     \beta . n
        //                                     * JxW[point];                  //
        //                                     dx
      }
  };

  auto face_worker = [&](const Iterator &    cell,
                         const unsigned int &f,
                         const unsigned int &sf,
                         const Iterator &    ncell,
                         const unsigned int &nf,
                         const unsigned int &nsf,
                         ScratchData<dim> &  scratch_data,
                         CopyData &          copy_data) {
    FEInterfaceValues<dim> &fe_iv = scratch_data.fe_interface_values;
    fe_iv.reinit(cell, f, sf, ncell, nf, nsf);
    const auto &q_points = fe_iv.get_quadrature_points();

    copy_data.face_data.emplace_back();
    CopyDataFace &copy_data_face = copy_data.face_data.back();

    const unsigned int n_dofs        = fe_iv.n_current_interface_dofs();
    copy_data_face.joint_dof_indices = fe_iv.get_interface_dof_indices();

    copy_data_face.cell_matrix.reinit(n_dofs, n_dofs);

    const std::vector<double> &        JxW     = fe_iv.get_JxW_values();
    const std::vector<Tensor<1, dim>> &normals = fe_iv.get_normal_vectors();

    for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
      {
        //        const double beta_dot_n = beta(q_points[qpoint]) *
        //        normals[qpoint]; for (unsigned int i = 0; i < n_dofs; ++i)
        //          for (unsigned int j = 0; j < n_dofs; ++j)
        //            copy_data_face.cell_matrix(i, j) +=
        //              fe_iv.jump(i, qpoint)                            //
        //              [\phi_i]
        //              * fe_iv.shape_value((beta_dot_n > 0), j, qpoint) //
        //              phi_j^{upwind}
        //              * beta_dot_n                                     //
        //              (\beta . n)
        //              * JxW[qpoint];                                   // dx
      }
  };

  AffineConstraints<double> constraints;

  auto copier = [&](const CopyData &c) {
    constraints.distribute_local_to_global(c.cell_matrix,
                                           c.cell_rhs,
                                           c.local_dof_indices,
                                           system_matrix,
                                           system_rhs);

    for (auto &cdf : c.face_data)
      {
        constraints.distribute_local_to_global(cdf.cell_matrix,
                                               cdf.joint_dof_indices,
                                               system_matrix);
      }
  };

  const unsigned int n_gauss_points = dof_handler.get_fe().degree + 1;

  ScratchData<dim> scratch_data(mapping, fe, n_gauss_points);
  CopyData         copy_data;

  MeshWorker::mesh_loop(dof_handler.begin_active(),
                        dof_handler.end(),
                        cell_worker,
                        copier,
                        scratch_data,
                        copy_data,
                        MeshWorker::assemble_own_cells |
                          MeshWorker::assemble_boundary_faces |
                          MeshWorker::assemble_own_interior_faces_once,
                        boundary_worker,
                        face_worker);
}

template <int dim>
void
DGHeat<dim>::assemble_system_CG()
{
  QGauss<dim> quadrature_formula(5);

  RightHandSideMMS<dim> right_hand_side;

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();

  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              // Stiffness Matrix
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) *
                 fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index));

            if (simulationCase_ == MMS)
              {
                // Right Hand Side
                cell_rhs(i) +=
                  (fe_values.shape_value(i, q_index) *
                   right_hand_side.value(fe_values.quadrature_point(q_index)) *
                   fe_values.JxW(q_index));
              }
          }


      // Assemble global matrix
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }


  std::map<types::global_dof_index, double> boundary_values;
  if (simulationCase_ == TaylorCouette)
    {
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(),
                                               boundary_values);
      VectorTools::interpolate_boundary_values(
        dof_handler, 1, Functions::ConstantFunction<dim>(1.), boundary_values);
    }

  if (simulationCase_ == MMS)
    {
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(),
                                               boundary_values);
    }

  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}

template <int dim>
void
DGHeat<dim>::solve()
{
  SolverControl solver_control(10000, 1e-12);
  SolverCG<>    solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  // We have made one addition, though: since we suppress output from the
  // linear solvers, we have to print the number of iterations by hand.
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}

template <int dim>
void
DGHeat<dim>::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  std::string dimension(dim == 2 ? "solution-2d-case-" : "solution-3d-case-");

  std::string fname = dimension + Utilities::int_to_string(simulationCase_) +
                      "-" + Utilities::int_to_string(refinementLevel_) + ".vtk";

  std::ofstream output(fname.c_str());
  data_out.write_vtk(output);
}

// Find the l2 norm of the error between the finite element sol'n and the exact
// sol'n
template <int dim>
void
DGHeat<dim>::calculateL2Error()
{
  QGauss<dim>   quadrature_formula(5);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell =
    fe.dofs_per_cell; // This gives you dofs per cell
  std::vector<types::global_dof_index> local_dof_indices(
    dofs_per_cell); //  Local connectivity

  const unsigned int n_q_points = quadrature_formula.size();

  double l2error = 0.;

  // loop over elements
  typename DoFHandler<dim>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);

      // Retrieve the effective "connectivity matrix" for this element
      cell->get_dof_indices(local_dof_indices);


      for (unsigned int q = 0; q < n_q_points; q++)
        {
          const double x = fe_values.quadrature_point(q)[0];
          const double y = fe_values.quadrature_point(q)[1];
          if (dim > 2)
            const double z = fe_values.quadrature_point(q)[2];

          const double r       = std::sqrt(x * x + y * y);
          const double lnratio = std::log(1. / 0.25);
          double       u_exact = 0.;
          if (simulationCase_ == TaylorCouette)
            u_exact = 1. / (lnratio)*std::log(r / 0.25);
          if (simulationCase_ == MMS)
            u_exact = -sin(M_PI * x) * std::sin(M_PI * y);
          double u_sim = 0;

          // Find the values of x and u_h (the finite element solution) at the
          // quadrature points
          for (unsigned int i = 0; i < dofs_per_cell; i++)
            {
              u_sim +=
                fe_values.shape_value(i, q) * solution[local_dof_indices[i]];
            }
          l2error += (u_sim - u_exact) * (u_sim - u_exact) * fe_values.JxW(q);
          //       std::cout << " x = " << x << " y = " << y <<  " r = " << r <<
          //       "   u_exact = " << u_exact << "   u_sim=" << u_sim <<
          //       std::endl;
        }
    }
  std::cout << "L2Error is : " << std::sqrt(l2error) << std::endl;
  L2Error_ = std::sqrt(l2error);
}



template <int dim>
void
DGHeat<dim>::run()
{
  make_grid();
  setup_system();
  assemble_system_CG();
  solve();
  output_results();
  calculateL2Error();
}

int
main()
{
  deallog.depth_console(0);

  // Taylor couette
  {
    int                 nsim = 5;
    std::vector<double> l2error;
    std::vector<int>    size;
    for (int m = 1; m < 1 + nsim; ++m)
      {
        std::cout << "Solving Taylor-Couette problem 2D - "
                  << " with mesh - " << m << std::endl;
        DGHeat<2> taylorCouette_problem_2d(TaylorCouette, m);
        taylorCouette_problem_2d.run();
        l2error.push_back(taylorCouette_problem_2d.getL2Error());
        size.push_back(m);
      }
    std::ofstream output_file("./L2Error-TaylorCouette.dat");
    for (unsigned int i = 0; i < size.size(); ++i)
      {
        output_file << size[i] << " " << l2error[i] << std::endl;
      }
  }

  // MMS
  {
    int                 nsim = 8;
    std::vector<double> l2error;
    std::vector<int>    size;
    for (int m = 2; m < 1 + nsim; ++m)
      {
        std::cout << "Solving MMS problem 2D - "
                  << " with mesh - " << m << std::endl;
        DGHeat<2> mms_problem_2d(MMS, m);
        mms_problem_2d.run();
        l2error.push_back(mms_problem_2d.getL2Error());
        size.push_back(m);
      }
    std::ofstream output_file("./L2Error-MMS.dat");
    for (unsigned int i = 0; i < size.size(); ++i)
      {
        output_file << size[i] << " " << l2error[i] << std::endl;
      }
  }

  return 0;
}
