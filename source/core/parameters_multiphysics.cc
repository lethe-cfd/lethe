#include <core/parameters_multiphysics.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

DeclException1(
  SharpeningThresholdError,
  double,
  << "Sharpening threshold : " << arg1 << " is smaller than 0 or larger than 1."
  << " Interface sharpening model requires a sharpening threshold between 0 and 1.");

DeclException1(
  SharpeningFrequencyError,
  int,
  << "Sharpening frequency : " << arg1 << " is equal or smaller than 0."
  << " Interface sharpening model requires an integer sharpening frequency larger than 0.");


void
Parameters::Multiphysics::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("multiphysics");
  {
    prm.declare_entry("fluid dynamics",
                      "true",
                      Patterns::Bool(),
                      "Fluid flow calculation <true|false>");

    prm.declare_entry("heat transfer",
                      "false",
                      Patterns::Bool(),
                      "Thermic calculation <true|false>");

    prm.declare_entry("tracer",
                      "false",
                      Patterns::Bool(),
                      "Passive tracer calculation <true|false>");

    prm.declare_entry("VOF",
                      "false",
                      Patterns::Bool(),
                      "VOF calculation <true|false>");

    // subparameters for heat_transfer
    prm.declare_entry("viscous dissipation",
                      "false",
                      Patterns::Bool(),
                      "Viscous dissipation in heat equation <true|false>");

    prm.declare_entry("buoyancy force",
                      "false",
                      Patterns::Bool(),
                      "Buoyant force calculation <true|false>");
  }
  prm.leave_subsection();

  vof_parameters.declare_parameters(prm);
}

void
Parameters::Multiphysics::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("multiphysics");
  {
    fluid_dynamics = prm.get_bool("fluid dynamics");
    heat_transfer  = prm.get_bool("heat transfer");
    tracer         = prm.get_bool("tracer");
    VOF            = prm.get_bool("VOF");

    // subparameter for heat_transfer
    viscous_dissipation = prm.get_bool("viscous dissipation");
    buoyancy_force      = prm.get_bool("buoyancy force");
  }
  prm.leave_subsection();

  vof_parameters.parse_parameters(prm);
}

void
Parameters::VOF::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("VOF");
  {
    prm.declare_entry("continuum surface force",
                      "false",
                      Patterns::Bool(),
                      "Continuum surface force calculation <true|false>");

    prm.declare_entry(
      "peeling wetting",
      "false",
      Patterns::Bool(),
      "Enable peeling/wetting in free surface calculation <true|false>");

    prm.declare_entry(
      "diffusion",
      "0",
      Patterns::Double(),
      "Diffusion value in the VOF transport equation. "
      "Default value is 0 to have pure advection. Use this parameter, "
      "along with interface sharpening, for peeling-wetting");

    conservation.declare_parameters(prm);
    sharpening.declare_parameters(prm);
  }
  prm.leave_subsection();
}

void
Parameters::VOF::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("VOF");
  {
    continuum_surface_force = prm.get_bool("continuum surface force");
    peeling_wetting         = prm.get_bool("peeling wetting");
    diffusion               = prm.get_double("diffusion");

    conservation.parse_parameters(prm);
    sharpening.parse_parameters(prm);
  }
  prm.leave_subsection();
}

void
Parameters::VOF_MassConservation::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("mass conservation");
  {
    prm.declare_entry(
      "skip mass conservation in fluid 0",
      "false",
      Patterns::Bool(),
      "Enable skipping mass conservation in fluid 0 <true|false>");

    prm.declare_entry(
      "skip mass conservation in fluid 1",
      "false",
      Patterns::Bool(),
      "Enable skipping mass conservation in fluid 1 <true|false>");

    prm.declare_entry(
      "monitoring",
      "false",
      Patterns::Bool(),
      "Enable conservation monitoring in free surface calculation <true|false>");

    prm.declare_entry(
      "fluid monitored",
      "1",
      Patterns::Integer(),
      "Index of the fluid which conservation is monitored <0|1>");
  }
  prm.leave_subsection();
}

void
Parameters::VOF_MassConservation::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("mass conservation");
  {
    skip_mass_conservation_fluid_0 =
      prm.get_bool("skip mass conservation in fluid 0");
    skip_mass_conservation_fluid_1 =
      prm.get_bool("skip mass conservation in fluid 1");
    monitoring         = prm.get_bool("monitoring");
    id_fluid_monitored = prm.get_integer("fluid monitored");
  }
  prm.leave_subsection();
}

void
Parameters::VOF_InterfaceSharpening::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("interface sharpening");
  {
    prm.declare_entry("interface sharpening",
                      "false",
                      Patterns::Bool(),
                      "Interface sharpening <true|false>");

    prm.declare_entry(
      "sharpening threshold",
      "0.5",
      Patterns::Double(),
      "VOF interface sharpening threshold that represents the mass conservation level");

    // This parameter must be larger than 1 for interface sharpening. Choosing
    // values less than 1 leads to interface smoothing instead of sharpening.
    prm.declare_entry(
      "interface sharpness",
      "2",
      Patterns::Double(),
      "Sharpness of the moving interface (parameter alpha in the interface sharpening model)");
    prm.declare_entry("sharpening frequency",
                      "10",
                      Patterns::Integer(),
                      "VOF interface sharpening frequency");
    prm.declare_entry(
      "verbosity",
      "quiet",
      Patterns::Selection("quiet|verbose"),
      "State whether from the interface sharpening calculations should be printed "
      "Choices are <quiet|verbose>.");
  }
  prm.leave_subsection();
}

void
Parameters::VOF_InterfaceSharpening::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("interface sharpening");
  {
    interface_sharpening = prm.get_bool("interface sharpening");
    sharpening_threshold = prm.get_double("sharpening threshold");
    interface_sharpness  = prm.get_double("interface sharpness");
    sharpening_frequency = prm.get_integer("sharpening frequency");

    Assert(sharpening_threshold > 0.0 && sharpening_threshold < 1.0,
           SharpeningThresholdError(sharpening_threshold));

    Assert(sharpening_frequency > 0,
           SharpeningFrequencyError(sharpening_frequency));

    const std::string op = prm.get("verbosity");
    if (op == "verbose")
      verbosity = Parameters::Verbosity::verbose;
    else if (op == "quiet")
      verbosity = Parameters::Verbosity::quiet;
    else
      throw(std::runtime_error("Invalid verbosity level"));
  }
  prm.leave_subsection();
}
