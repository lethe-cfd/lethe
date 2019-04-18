#include "glsNS.h"

template <int dim>
class MMSUnstructuredNavierStokes : public GLSNavierStokesSolver<dim>
{
public:
  MMSUnstructuredNavierStokes(const std::string input_filename, const unsigned int degreeVelocity, const unsigned int degreePressure);
  void runMMSUnstructured();

private:

  std::vector<double>          wallTime_;
};

template <int dim>
MMSUnstructuredNavierStokes<dim>::MMSUnstructuredNavierStokes(const std::string input_filename, const unsigned int degreeVelocity, const unsigned int degreePressure):
  GLSNavierStokesSolver<dim>(input_filename,degreeVelocity,degreePressure)
{

}


template<int dim>
void MMSUnstructuredNavierStokes<dim>::runMMSUnstructured()
{
  std::vector<double>                   ErrorLog;
  std::vector<double>                   wallTime;
  GridIn<dim> grid_in;
  grid_in.attach_triangulation (this->triangulation);
  std::ifstream input_file(this->meshParameters.fileName);
  grid_in.read_msh(input_file);
  this->setup_dofs();
  this->exact_solution = new ExactSolutionMMS<dim>;
  this->forcing_function = new MMSSineForcingFunction<dim>;
  this->viscosity_=this->physicalProperties.viscosity;

  Timer timer;
  while(this->simulationControl.integrate())
  {
    printTime(this->pcout,this->simulationControl);
    timer.start ();
    if (this->simulationControl.getIter() !=1) this->refine_mesh();
    // Force restart from scratch of the solution
    this->newton_iteration(true);
    this->postprocess();
    double L2Error= this->calculateL2Error();
    this->pcout << "L2Error U is : " << std::setprecision(this->analyticalSolutionParameters.errorPrecision) << L2Error << std::endl;
    ErrorLog.push_back(L2Error);

    wallTime.push_back((timer.wall_time()));
    this->finishTimeStep();
  }

  if(this->this_mpi_process==0)
  {
    assert (wallTime.size()==ErrorLog.size());
    std::ofstream output_file("./L2Error-2D.dat");
    for (unsigned int i=0 ; i < ErrorLog.size() ; ++i)
    {
      output_file << i << " " << ErrorLog[i] << " " << wallTime[i] << std::endl;
    }
    output_file.close();
  }
}

int main (int argc, char *argv[])
{
    try
    {
    if (argc != 2)
      {
        std::cout << "Usage:" << argv[0] << " input_file" << std::endl;
        std::exit(1);
      }
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);
        Parameters::FEM              fem;
        fem=Parameters::getFEMParameters2D(argv[1]);
        MMSUnstructuredNavierStokes<2> problem_2d(argv[1],fem.velocityOrder,fem.pressureOrder);
        problem_2d.runMMSUnstructured();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
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
        std::cerr << std::endl << std::endl
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
