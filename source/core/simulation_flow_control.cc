#include "core/simulation_flow_control.h"

#include <fstream>

#include "core/parameters.h"


SimulationControl::SimulationControl(Parameters::SimulationControl param)
  : current_time(0)
  , time_step(param.dt)
  , end_time(param.timeEnd)
  , iteration_number(0)
  , number_mesh_adapt(param.number_mesh_adaptation)
  , CFL(0)
  , max_CFL(param.maxCFL)
  , output_frequency(param.output_frequency)
  , output_time_frequency(param.output_time)
  , subdivision(param.subdivision)
  , group_files(param.group_files)
  , output_name(param.output_name)
  , output_path(param.output_folder)
{
  time_step_vector.resize(numberTimeStepStored);
  time_step_vector[0] = param.dt;
}

void
SimulationControl::add_time_step(double p_timestep)
{
  time_step = p_timestep;
  // Store previous time step in table
  for (unsigned int i_time = time_step_vector.size() - 1; i_time > 0; --i_time)
    time_step_vector[i_time] = time_step_vector[i_time - 1];

  // Calculate time step, right now this is a dummy function
  time_step_vector[0] = p_timestep;
}

bool
SimulationControl::is_output_iteration()
{
  return (get_step_number() % output_frequency == 0);
}

void
SimulationControl::save(std::string prefix)
{
  std::string   filename = prefix + ".simulationcontrol";
  std::ofstream output(filename.c_str());
  output << "Simulation control" << std::endl;
  for (unsigned int i = 0; i < time_step_vector.size(); ++i)
    output << "dt_" << i << " " << time_step_vector[i] << std::endl;
  output << "CFL  " << CFL << std::endl;
  output << "Time " << current_time << std::endl;
  output << "Iter " << iteration_number << std::endl;
}

void
SimulationControl::read(std::string prefix)
{
  std::string   filename = prefix + ".simulationcontrol";
  std::ifstream input(filename.c_str());
  if (!input)
    {
      throw("Unable to open file");
    }
  std::string buffer;
  std::getline(input, buffer);
  for (unsigned int i = 0; i < time_step_vector.size(); ++i)
    input >> buffer >> time_step_vector[i];
  input >> buffer >> CFL;
  input >> buffer >> current_time;
  input >> buffer >> iteration_number;
}


SimulationControlTransient::SimulationControlTransient(
  Parameters::SimulationControl param)
  : SimulationControl(param)
  , adapt(param.adapt)
  , adaptative_time_step_scaling(param.adaptative_time_step_scaling)
{}

void
SimulationControlTransient::print_progression(const ConditionalOStream &pcout)
{
  pcout << std::endl;
  pcout << "*****************************************************************"
        << std::endl;
  pcout << "Transient iteration : " << std::setw(8) << std::left
        << iteration_number << " Time : " << std::setw(8) << std::left
        << current_time << " Time step : " << std::setw(8) << std::left
        << time_step << " CFL : " << std::setw(8) << std::left
        << SimulationControl::get_CFL() << std::endl;
  pcout << "*****************************************************************"
        << std::endl;
}

bool
SimulationControlTransient::integrate()
{
  if (!is_at_end())
    {
      iteration_number++;
      add_time_step(calculate_time_step());
      current_time += time_step;
      return true;
    }

  else
    return false;
}



bool
SimulationControlTransient::is_at_end()
{
  return current_time >= (end_time - 1e-12 * time_step);
}

double
SimulationControlTransient::calculate_time_step()
{
  double new_time_step = time_step;

  if (adapt && iteration_number > 1)
    {
      new_time_step = time_step * adaptative_time_step_scaling;
      if (CFL > 0 && max_CFL / CFL < adaptative_time_step_scaling)
        new_time_step = time_step * max_CFL / CFL;
    }
  if (current_time + new_time_step > end_time)
    new_time_step = end_time - current_time;

  return new_time_step;
}


SimulationControlTransientDynamicOutput::
  SimulationControlTransientDynamicOutput(Parameters::SimulationControl param)
  : SimulationControlTransient(param)
  , time_step_forced_output(false)
  , last_output_time(0.)
{}

double
SimulationControlTransientDynamicOutput::calculate_time_step()
{
  double new_time_step = time_step;
  if (time_step_forced_output)
    {
      new_time_step           = time_step_vector[1];
      time_step_forced_output = false;
    }
  else if (iteration_number > 1)
    {
      new_time_step = time_step * adaptative_time_step_scaling;
      if (CFL > 0 && max_CFL / CFL < adaptative_time_step_scaling)
        new_time_step = time_step * max_CFL / CFL;
    }

  if (current_time + new_time_step > end_time)
    new_time_step = end_time - current_time;

  if (current_time + new_time_step > last_output_time + output_time_frequency)
    {
      new_time_step = last_output_time + output_time_frequency - current_time;
      time_step_forced_output = true;
    }

  return new_time_step;
}

bool
SimulationControlTransientDynamicOutput::is_output_iteration()
{
  bool is_output_time =
    (current_time - last_output_time) - output_time_frequency >
    -1e-12 * output_time_frequency;
  if (is_output_time)
    last_output_time = current_time;

  return is_output_time;
}


SimulationControlSteady::SimulationControlSteady(
  Parameters::SimulationControl param)
  : SimulationControl(param)
{}

bool
SimulationControlSteady::integrate()
{
  if (!is_at_end())
    {
      iteration_number++;
      return true;
    }

  else
    return false;
}

void
SimulationControlSteady::print_progression(const ConditionalOStream &pcout)
{
  pcout << std::endl;
  pcout << "*****************************************************************"
        << std::endl;
  pcout << "Steady iteration : " << std::setw(8) << std::right
        << iteration_number << "/" << number_mesh_adapt + 1 << std::endl;
  pcout << "*****************************************************************"
        << std::endl;
}

bool
SimulationControlSteady::is_at_end()
{
  return iteration_number >= (number_mesh_adapt + 1);
}
