#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
class AdvectionField : public TensorFunction<1, dim>
{
public:
  virtual Tensor<1, dim> value(const Point<dim> &p) const override;
};


template <int dim>
Tensor<1, dim> AdvectionField<dim>::value(const Point<dim> &p) const
{
  Tensor<1, dim> value;
  value[0] = 1;
  for (unsigned int i = 1; i < dim; ++i)
    value[i] = 0;

  return value;
}

template <int dim>
class InitialConditions : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                       const unsigned int component = 0) const override;

private:
  static const Point<dim> center_point;
};

template <int dim>
double InitialConditions<dim>::value(const Point<dim>  &p,
                                  const unsigned int component) const
{
  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  
  Point<dim> center = Point<dim>();
  Tensor<1,dim> dist = center - p;
  if (dist.norm() > 0.25)
  {
    return 0.0;
  }
  return 1.0;
  
  // if (p[0]> 0.25 || p[0]<-0.25)
  //   return 0.0;
  // return 0.5 + 0.5*std::cos(4*p[0]*M_PI);
}

template <int dim>
class AdvectionProblem
{
public:
  AdvectionProblem();
  void run();

private:
  
  using VectorType = TrilinosWrappers::MPI::Vector;
  using MatrixType = TrilinosWrappers::SparseMatrix;
  
  void setup_system();
  
  struct AssemblyScratchData
  {
    // Constructors
    AssemblyScratchData(const FiniteElement<dim> &fe);
    AssemblyScratchData(const AssemblyScratchData &scratch_data);
    
    // Set up FEValues to reuse them because their initialization is expensive.
    FEValues<dim>     fe_values;
    
    // Previous phase values
    std::vector<double>    previous_phase_values;
    
    std::vector<Tensor<1, dim>> advection_directions;
    
    // Velocity
    AdvectionField<dim> advection_field;
  };

  struct AssemblyCopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
  };
  
  void assemble_system();
  void local_assemble_system(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    AssemblyScratchData                                  &scratch,
    AssemblyCopyData                                     &copy_data);    
  void copy_local_to_global(const AssemblyCopyData &copy_data);
  
  void set_initial_conditions();
  void solve();
  
  void compute_level_set_from_phase_fraction();
  void compute_phase_fraction_from_level_set();
  void identify_cell_location();
  
  
  void refine_grid();
  void output_results(const int time_iteration, const double time) const;
  
  parallel::distributed::Triangulation<dim> triangulation;
  const MappingQ<dim>                       mapping;
  
  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;
  MatrixType                system_matrix;
  
  VectorType level_set;
  
  NonMatching::MeshClassifier<dim> mesh_classifier;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  
  VectorType solution;
  VectorType previous_solution;
  VectorType location;
  
  VectorType system_rhs;
  
  double dt;
  
  MPI_Comm           mpi_communicator;
  ConditionalOStream pcout;
};

template <int dim>
AdvectionProblem<dim>::AdvectionProblem()
  : triangulation(MPI_COMM_WORLD,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , mapping(2)
  , fe(1)
  , dof_handler(triangulation)
  , mesh_classifier(dof_handler, level_set)
  , dt(0.001)
  , mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{}
  
template <int dim>
void AdvectionProblem<dim>::setup_system()
{
  pcout << "boop 0" << std::endl;
  dof_handler.distribute_dofs(fe);
  
  pcout << "boop 1" << std::endl;
  
  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  
  pcout << "boop 2" << std::endl;
  
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  previous_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  level_set.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  location.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  
  
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
                     
  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler,
                                          constraints);
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs =*/false);
  
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);

}  
  
template <int dim>
void AdvectionProblem<dim>::assemble_system()
{
  system_matrix = 0;
  system_rhs = 0;
  
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &AdvectionProblem::local_assemble_system,
                  &AdvectionProblem::copy_local_to_global,
                  AssemblyScratchData(fe),
                  AssemblyCopyData());
}      

template <int dim>
AdvectionProblem<dim>::AssemblyScratchData::AssemblyScratchData(
  const FiniteElement<dim> &fe)
  : fe_values(fe,
              QGauss<dim>(fe.degree + 1),
              update_values | update_gradients | update_quadrature_points |
                update_JxW_values)
  , advection_directions(fe_values.get_quadrature().size())
{
  const unsigned int n_q_points =
    fe_values.get_quadrature().size();
    
  this->previous_phase_values = std::vector<double>(n_q_points);
}

template <int dim>
AdvectionProblem<dim>::AssemblyScratchData::AssemblyScratchData(
  const AssemblyScratchData &scratch_data)
  : fe_values(scratch_data.fe_values.get_fe(),
              scratch_data.fe_values.get_quadrature(),
              update_values | update_gradients | update_quadrature_points |
                update_JxW_values)
  , advection_directions(fe_values.get_quadrature().size())
{
  const unsigned int n_q_points =
    fe_values.get_quadrature().size();
    
  this->previous_phase_values = std::vector<double>(n_q_points);
}

template <int dim>
void AdvectionProblem<dim>::local_assemble_system(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  AssemblyScratchData                                  &scratch_data,
  AssemblyCopyData                                     &copy_data)
{
  if (!cell->is_locally_owned())
    return;
  
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points =
    scratch_data.fe_values.get_quadrature().size();

  copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
  copy_data.cell_rhs.reinit(dofs_per_cell);
  
  copy_data.local_dof_indices.resize(dofs_per_cell);
  
  scratch_data.fe_values.reinit(cell);
  scratch_data.fe_values.get_function_values(previous_solution, scratch_data.previous_phase_values);
  
  scratch_data.advection_field.value_list(
    scratch_data.fe_values.get_quadrature_points(),
    scratch_data.advection_directions);
    
  double cell_size;
  
  if (dim == 2)
    {
      cell_size = std::sqrt(4. * cell->measure() / M_PI);
    }
  else if (dim == 3)
    {
      cell_size = std::pow(6 * cell->measure() / M_PI, 1. / 3.);
    }
  
  double dt_inv = 1.0/this->dt;
  
  const auto &sd = scratch_data;
  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    {
      const Tensor<1, dim> velocity = sd.advection_directions[q_point];
      
      const double u_mag = std::max(velocity.norm(), 1e-12);
      
      const double tau =   1. /
                std::sqrt(Utilities::fixed_power<2>(dt_inv) +
                          Utilities::fixed_power<2>(2. * u_mag / cell_size));
                          
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            // LHS
            
            // Time integration
            copy_data.cell_matrix(i, j) += dt_inv*sd.fe_values.shape_value(i, q_point)*sd.fe_values.shape_value(j, q_point)* sd.fe_values.JxW(q_point);
            // Advective term
            copy_data.cell_matrix(i, j) += sd.fe_values.shape_value(i, q_point)*sd.advection_directions[q_point]*sd.fe_values.shape_grad(j, q_point)*sd.fe_values.JxW(q_point);
            // Stabilization term
            
            copy_data.cell_matrix(i, j) += tau*sd.advection_directions[q_point]*sd.fe_values.shape_grad(i, q_point)*(sd.advection_directions[q_point]*sd.fe_values.shape_grad(j, q_point) + sd.fe_values.shape_value(j, q_point)*dt_inv)*sd.fe_values.JxW(q_point);
            
          }
          //RHS
          
          // Stabilization term
          copy_data.cell_rhs(i) += tau*sd.advection_directions[q_point]*sd.fe_values.shape_grad(i, q_point)*dt_inv*scratch_data.previous_phase_values[q_point]*sd.fe_values.JxW(q_point);
          
          copy_data.cell_rhs(i) += dt_inv*sd.fe_values.shape_value(i, q_point)*scratch_data.previous_phase_values[q_point]*sd.fe_values.JxW(q_point);
      }
      
    }
    cell->get_dof_indices(copy_data.local_dof_indices);
    
}

template <int dim>
void
AdvectionProblem<dim>::copy_local_to_global(const AssemblyCopyData &copy_data)
{
  constraints.distribute_local_to_global(
    copy_data.cell_matrix,
    copy_data.cell_rhs,
    copy_data.local_dof_indices,
    system_matrix,
    system_rhs);
}

template <int dim>
void
AdvectionProblem<dim>::set_initial_conditions()
{
  VectorTools::interpolate(this->dof_handler,
                           InitialConditions<dim>(),
                           this->previous_solution);
  this->solution = this->previous_solution;                 
}

template <int dim>
void AdvectionProblem<dim>::solve()
{
  SolverControl               solver_control(1000,
                               1e-10 * system_rhs.l2_norm() );
  TrilinosWrappers::SolverGMRES solver(solver_control);
  
  TrilinosWrappers::PreconditionILU                 preconditioner;
  TrilinosWrappers::PreconditionILU::AdditionalData data_ilu;
  
  preconditioner.initialize(system_matrix, data_ilu);

  solver.solve(system_matrix,
              solution,
              system_rhs,
              preconditioner);

  VectorType residual(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  
  system_matrix.vmult(residual, solution);
  // residual -= system_rhs;
  pcout << "   Iterations required for convergence: "
            << solver_control.last_step() << '\n'
            << "   Max norm of residual:                "
            << residual.linfty_norm() << '\n';

  constraints.distribute(solution);
}

template <int dim>
void 
AdvectionProblem<dim>::compute_level_set_from_phase_fraction()
{
  VectorType level_set_owned(this->locally_owned_dofs,
                                           mpi_communicator);
  for (auto p : this->locally_owned_dofs)
    {
      // level_set_owned[p] = std::log(std::max((2.0*solution[p])/(2.0-solution[p]),1e-12));
      level_set_owned[p] = 2.0*solution[p] - 1.0;
    }
  level_set = level_set_owned;  
}

template <int dim>
void 
AdvectionProblem<dim>::identify_cell_location()
{  
  
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      
      if (!cell->is_locally_owned())
        return;
      
      const NonMatching::LocationToLevelSet cell_location =
            mesh_classifier.location_to_level_set(cell);
      
      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
      
      
      std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
      
      cell->get_dof_indices(dof_indices);
      
      
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        if (cell_location == NonMatching::LocationToLevelSet::intersected)
          location[dof_indices[i]] = 1.0;
      }
      
      // if (cell_location == NonMatching::LocationToLevelSet::outside)
      // {
      // 
      // }
        
    }
}

template <int dim>
void AdvectionProblem<dim>::output_results(const int time_iteration, const double time) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(level_set, "level_set");
  data_out.add_data_vector(location, "location");
  
  
  data_out.build_patches();

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(vtk_flags);

  const std::string filename = "solution.vtu";
  // std::ofstream     output(filename);
  // data_out.write_vtu(output);
  data_out.write_vtu_with_pvtu_record("output/",
                                      "solution",
                                      time_iteration,
                                      MPI_COMM_WORLD,
                                      3);
  // pcout << "Solution written to " << filename << std::endl;
  std::ofstream out("grid-2.svg");
  GridOut       grid_out;
    grid_out.write_svg(triangulation, out);
}

template <int dim>
void AdvectionProblem<dim>::run()
{
  pcout << "Bonjour from run" << std::endl;
  
  Point<dim> p_0 = Point<dim>();
  p_0[0] = -1;
  for (unsigned int i = 1; i < dim; ++i)
    p_0[i] = -1;
    
  Point<dim> p_1 = Point<dim>();
  p_1[0] = 1;
  for (unsigned int i = 1; i < dim; ++i)
    p_1[i] = 1;
    
  std::vector< unsigned int > repetitions(dim);
  repetitions[0] = 1;
  for (unsigned int i = 1; i < dim; ++i)
    repetitions[i] = 1;
    
  GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p_0, p_1);
  triangulation.refine_global(8);
            
  pcout << "Bonjour from after triangulation" << std::endl;
  // initial time step
  setup_system();
  
  pcout << "Bonjour from after setup_system()" << std::endl;
  
  
  set_initial_conditions();
  
  
  unsigned int it = 0;
  double time = 0.0;
  double final_time = 0.01;
  
  compute_level_set_from_phase_fraction();
  
  mesh_classifier.reclassify();

  
  identify_cell_location();
  
  output_results(it,time);
  
  
  pcout << "Bonjour from after set_initial_conditions()" << std::endl;
  
  while (time < final_time) 
  {
    it += 1;
    time += this->dt;
    
    
    
    assemble_system();
    
    pcout << "time = " << time << std::endl;

    solve();
    
    compute_level_set_from_phase_fraction();
    
    output_results(it, time);
    
    // 
    // previous_solution = solution;
    this->previous_solution = this->solution; 
  } 
  
  

}

int main(int argc, char *argv[])
{  
Utilities::MPI::MPI_InitFinalize       mpi_init(argc, argv, 1);
try
  {
    
    AdvectionProblem<2> advection_problem_2d;
    advection_problem_2d.run();
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


