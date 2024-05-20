#include "core/utilities.h"

#include <solvers/gd_navier_stokes.h>

#include <deal.II/base/convergence_table.h>

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

      const unsigned int                  dim = get_dimension(argv[1]);
      const Parameters::SizeOfSubsections size_of_subsections =
        Parameters::get_size_of_subsections(argv[1]);

      if (dim == 2)
        {
          ParameterHandler        prm;
          SimulationParameters<2> NSparam;
          NSparam.declare(prm, size_of_subsections);
          // Parsing of the file
          prm.parse_input(argv[1]);
          NSparam.parse(prm);

          AssertThrow(NSparam.nitsche->number_solids == 0,
                      SolidWarning(NSparam.nitsche->number_solids,
                                   "gd_navier_stokes_2d",
                                   "gls_nitsche_navier_stokes_22"));

          GDNavierStokesSolver<2> problem(NSparam);
          problem.solve();
        }

      else if (dim == 3)
        {
          ParameterHandler        prm;
          SimulationParameters<3> NSparam;
          NSparam.declare(prm, size_of_subsections);
          // Parsing of the file
          prm.parse_input(argv[1]);
          NSparam.parse(prm);

          AssertThrow(NSparam.nitsche->number_solids == 0,
                      SolidWarning(NSparam.nitsche->number_solids,
                                   "gd_navier_stokes_2d",
                                   "gls_nitsche_navier_stokes_22"));

          GDNavierStokesSolver<3> problem(NSparam);
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
