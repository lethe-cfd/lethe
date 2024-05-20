/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - by the Lethe authors
 *
 * This file is part of the Lethe library
 *
 * The Lethe library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 3.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the Lethe distribution.
 *
 * ---------------------------------------------------------------------
 */

#include "core/utilities.h"
#include "solvers/gls_nitsche_navier_stokes.h"

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

          AssertThrow(NSparam.nitsche->number_solids > 0,
                      NoSolidWarning("gls_nitsche_navier_stokes_22"));

          GLSNitscheNavierStokesSolver<2> problem_22(NSparam);
          problem_22.solve();
        }

      else if (dim == 3)
        {
          ParameterHandler        prm;
          SimulationParameters<3> NSparam;
          NSparam.declare(prm, size_of_subsections);
          // Parsing of the file
          prm.parse_input(argv[1]);
          NSparam.parse(prm);

          AssertThrow(NSparam.nitsche->number_solids > 0,
                      NoSolidWarning("gls_nitsche_navier_stokes_22"));

          GLSNitscheNavierStokesSolver<3> problem_33(NSparam);
          problem_33.solve();
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
