#include "core/utilities.h"
#include <rpt/rpt_calculating_parameters.h>
#include <rpt/rpt_fem_reconstruction.h>

#include <fstream>
#include <iostream>

using namespace dealii;

int
main(int argc, char *argv[])
{
  try
    {
      if (argc != 2)
        {
          std::cout << "Usage:" << argv[0] << " input_file" << std::endl;
          std::exit(1);
        }

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      ParameterHandler         prm;
      RPTCalculatingParameters rpt_parameters;
      rpt_parameters.declare(prm);

      // Parsing of the file
      prm.parse_input(argv[1]);
      rpt_parameters.parse(prm);

      RPTL2Projection<3> rpt_l2_project(rpt_parameters.rpt_param,
                                        rpt_parameters.fem_reconstruction_param,
                                        rpt_parameters.detector_param);
      rpt_l2_project.L2_project();
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
