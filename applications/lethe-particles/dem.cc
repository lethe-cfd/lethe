#include "dem/dem.h"

#include "core/dem_properties.h"
#include "core/utilities.h"

using namespace dealii;

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);


      if (argc != 2)
        {
          std::cout << "Usage:" << argv[0] << " input_file" << std::endl;
          std::exit(1);
        }

      const unsigned int dim = get_dimension(argv[1]);

      if (dim == 2)
        {
          ParameterHandler       prm;
          DEMSolverParameters<2> dem_parameters;
          dem_parameters.declare(prm);

          // Parsing of the file
          prm.parse_input(argv[1]);
          dem_parameters.parse(prm);

          DEMSolver<2> problem(dem_parameters);
          problem.solve();
        }

      else if (dim == 3)
        {
          ParameterHandler       prm;
          DEMSolverParameters<3> dem_parameters;
          dem_parameters.declare(prm);

          // Parsing of the file
          prm.parse_input(argv[1]);
          dem_parameters.parse(prm);

          DEMSolver<3> problem(dem_parameters);
          problem.solve();
        }

      else
        {
          return 1;
        }
    }
  catch (std::exception &exc)
    {
      announce_exception(exc);
    }
  catch (...)
    {
      announce_unknown_exception();
    }
  return 0;
}
