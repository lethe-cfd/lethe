﻿/*
 * contactForce.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: shahab
 */

#include "dem/contact_force.h"

#include <deal.II/base/point.h>

#include "dem/dem_iterator.h"

using namespace dealii;

template <int dim, int spacedim>
ContactForce<dim, spacedim>::ContactForce()
{}

template <int dim, int spacedim>
void
ContactForce<dim, spacedim>::linearCF(
  std::vector<
    std::tuple<std::pair<typename Particles::ParticleIterator<dim, spacedim>,
                         typename Particles::ParticleIterator<dim, spacedim>>,
               double,
               Point<dim>,
               double,
               Point<dim>,
               double,
               double>> contactInfo,
  ParametersDEM<dim>    DEMparam)
{
  for (unsigned int i = 0; i < contactInfo.size(); i++)
    {
      {
        Point<dim> totalForce;
        double     mEff =
          (std::get<0>(contactInfo[i]).first->get_properties()[19] *
           std::get<0>(contactInfo[i]).second->get_properties()[19]) /
          (std::get<0>(contactInfo[i]).first->get_properties()[19] +
           std::get<0>(contactInfo[i]).second->get_properties()[19]);
        double rEff =
          (std::get<0>(contactInfo[i]).first->get_properties()[2] *
           std::get<0>(contactInfo[i]).second->get_properties()[2]) /
          (2.0 * (std::get<0>(contactInfo[i]).first->get_properties()[2] +
                std::get<0>(contactInfo[i]).second->get_properties()[2]));
        double yEff = DEMparam.physicalProperties.Yp /
                      (2.0 * (1.0 - pow(DEMparam.physicalProperties.vp, 2.0)));
        // double gEff = (DEMparam.physicalProperties.Yp) / (4 *
        // (2-DEMparam.physicalProperties.vp) *
        // (1+DEMparam.physicalProperties.vp));
        double kn = 1.2024 * pow((pow(mEff, 0.5) * pow(yEff, 2.0) * rEff *
                                  abs(std::get<3>(contactInfo[i]))),
                                 0.4);
        double kt = 1.2024 * pow((pow(mEff, 0.5) * pow(yEff, 2.0) * rEff *
                                  abs(std::get<5>(contactInfo[i]))),
                                 0.4);
        double ethan =
          (-2.0 * log(DEMparam.physicalProperties.ep) * sqrt(mEff * kn)) /
          (sqrt((pow(log(DEMparam.physicalProperties.ep), 2.0)) +
                pow(3.1415, 2.0)));
        double ethat = 0;
        if (DEMparam.physicalProperties.ep == 0)
          {
            ethat = 2.0 * sqrt(2.0 / 7.0 * mEff * kt);
          }
        else
          {
            ethat = (-2.0 * log(DEMparam.physicalProperties.ep) *
                     sqrt(2.0 / 7.0 * mEff * kt)) /
                    (sqrt(pow(3.1415, 2.0) +
                          pow(log(DEMparam.physicalProperties.ep), 2.0)));
          }

        Point<dim> normalForce = ((-1.0 * kn * std::get<1>(contactInfo[i])) -
                                  (ethan * std::get<3>(contactInfo[i]))) *
                                 std::get<2>(contactInfo[i]);

        Point<dim> tangForce = (-1.0 * kt * std::get<6>(contactInfo[i]) -
                                ethat * std::get<5>(contactInfo[i])) *
                               std::get<4>(contactInfo[i]);

double coulumbLimit = DEMparam.physicalProperties.mup * normalForce.norm();
        if (tangForce.norm() <=
            coulumbLimit)
          {
            totalForce = normalForce + tangForce;
          }
        else
          {
            Point<dim> coulumbTangForce =
              coulumbLimit *
              sgn(std::get<6>(contactInfo[i])) * std::get<4>(contactInfo[i]);

            totalForce = normalForce + coulumbTangForce;


          }




        std::get<0>(contactInfo[i]).first->get_properties()[13] = totalForce[0];
        std::get<0>(contactInfo[i]).first->get_properties()[14] = totalForce[1];
        std::get<0>(contactInfo[i]).first->get_properties()[15] = totalForce[2];

        std::get<0>(contactInfo[i]).second->get_properties()[13] =
          -1.0 * totalForce[0];
        std::get<0>(contactInfo[i]).second->get_properties()[14] =
          -1.0 * totalForce[1];
        std::get<0>(contactInfo[i]).second->get_properties()[15] =
          -1.0 * totalForce[2];

        // calculation of torque
        /*
         Point<dim> torqueTi;
         torqueTi = ((std::get<0>(contactInfo[i]).first->get_properties()[2])/2.0) * cross_product_3d( std::get<2>(contactInfo[i]) , totalForce);
         Point<dim> torqueTj;
         torqueTj = ((std::get<0>(contactInfo[i]).second->get_properties()[2])/2.0) * cross_product_3d( std::get<2>(contactInfo[i]) , -1.0*totalForce);
        Point<dim> omegai = {std::get<0>(contactInfo[i]).first->get_properties()[16] ,   std::get<0>(contactInfo[i]).first->get_properties()[17] ,   std::get<0>(contactInfo[i]).first->get_properties()[18]};
        Point<dim> omegaj = {std::get<0>(contactInfo[i]).second->get_properties()[16] ,   std::get<0>(contactInfo[i]).second->get_properties()[17] ,   std::get<0>(contactInfo[i]).second->get_properties()[18]};

        Point<dim> omegaij = {0.0, 0.0, 0.0};
        double omegaNorm = (omegai - omegaj).norm();
        if(omegaNorm != 0)
        {omegaij = (omegai - omegaj) / omegaNorm ;}
        Point<dim> torquer;
       torquer = -1.0 * DEMparam.physicalProperties.murp * rEff * normalForce.norm() * omegaij;
       std::get<0>(contactInfo[i]).first->get_properties()[21] = torqueTi[0] + torquer[0];
       std::get<0>(contactInfo[i]).first->get_properties()[22] = torqueTi[1] + torquer[1];
       std::get<0>(contactInfo[i]).first->get_properties()[23] = torqueTi[2] + torquer[2];
       std::get<0>(contactInfo[i]).second->get_properties()[21] = torqueTj[0] + torquer[0];
       std::get<0>(contactInfo[i]).second->get_properties()[22] = torqueTj[1] + torquer[1];
       std::get<0>(contactInfo[i]).second->get_properties()[23] = torqueTj[2] + torquer[2];
     */


      }
    }
}

template <int dim, int spacedim>
void
ContactForce<dim, spacedim>::nonLinearCF(
  std::vector<
    std::tuple<std::pair<typename Particles::ParticleIterator<dim, spacedim>,
                         typename Particles::ParticleIterator<dim, spacedim>>,
               double,
               Point<dim>,
               double,
               Point<dim>,
               double,
               double>> contactInfo,
  ParametersDEM<dim>    DEMparam)
{
  for (unsigned int i = 0; i < contactInfo.size(); i++)
    {
      {
        Point<dim> totalForce;
        double     mEff =
          (std::get<0>(contactInfo[i]).first->get_properties()[19] *
           std::get<0>(contactInfo[i]).second->get_properties()[19]) /
          (std::get<0>(contactInfo[i]).first->get_properties()[19] +
           std::get<0>(contactInfo[i]).second->get_properties()[19]);
        double rEff =
          (std::get<0>(contactInfo[i]).first->get_properties()[2] *
           std::get<0>(contactInfo[i]).second->get_properties()[2]) /
          (2.0 * (std::get<0>(contactInfo[i]).first->get_properties()[2] +
                std::get<0>(contactInfo[i]).second->get_properties()[2]));
        double yEff = DEMparam.physicalProperties.Yp /
                      (2.0 * (1.0 - pow(DEMparam.physicalProperties.vp, 2.0)));
        double gEff = (DEMparam.physicalProperties.Yp) / (4.0 *
         (2.0-DEMparam.physicalProperties.vp) *
         (1.0+DEMparam.physicalProperties.vp));
        double betha = log(DEMparam.physicalProperties.vp)/sqrt(pow(log(DEMparam.physicalProperties.vp),2.0) + 9.8696) ;
        double sn = 2.0 * yEff * sqrt(rEff * std::get<1>(contactInfo[i]));
        double st = 8.0 * gEff * sqrt(rEff * std::get<1>(contactInfo[i]));
        double kn = 1.3333 * yEff * sqrt(rEff * std::get<1>(contactInfo[i]));
        double ethan = -1.8257 * betha * sqrt(sn * mEff);
        double kt = 8.0 * gEff * sqrt(rEff * std::get<1>(contactInfo[i]));
        double ethat = -1.8257 * betha * sqrt(st * mEff);


        Point<dim> normalForce = ((-1.0 * kn * std::get<1>(contactInfo[i])) -
                                  (ethan * std::get<3>(contactInfo[i]))) *
                                 std::get<2>(contactInfo[i]);

        Point<dim> tangForce = (-1.0 * kt * std::get<6>(contactInfo[i]) -
                                ethat * std::get<5>(contactInfo[i])) *
                               std::get<4>(contactInfo[i]);

double coulumbLimit = DEMparam.physicalProperties.mup * normalForce.norm();
        if (tangForce.norm() <=
            coulumbLimit)
          {
            totalForce = normalForce + tangForce;
          }
        else
          {
            Point<dim> coulumbTangForce =
              coulumbLimit *
              sgn(std::get<6>(contactInfo[i])) * std::get<4>(contactInfo[i]);

            totalForce = normalForce + coulumbTangForce;


          }

        std::get<0>(contactInfo[i]).first->get_properties()[13] = totalForce[0];
        std::get<0>(contactInfo[i]).first->get_properties()[14] = totalForce[1];
        std::get<0>(contactInfo[i]).first->get_properties()[15] = totalForce[2];

        std::get<0>(contactInfo[i]).second->get_properties()[13] =
          -1.0 * totalForce[0];
        std::get<0>(contactInfo[i]).second->get_properties()[14] =
          -1.0 * totalForce[1];
        std::get<0>(contactInfo[i]).second->get_properties()[15] =
          -1.0 * totalForce[2];

        // calculation of torque
        /*
         Point<dim> torqueTi;
         torqueTi = ((std::get<0>(contactInfo[i]).first->get_properties()[2])/2.0) * cross_product_3d( std::get<2>(contactInfo[i]) , totalForce);
         Point<dim> torqueTj;
         torqueTj = ((std::get<0>(contactInfo[i]).second->get_properties()[2])/2.0) * cross_product_3d( std::get<2>(contactInfo[i]) , -1.0*totalForce);
        Point<dim> omegai = {std::get<0>(contactInfo[i]).first->get_properties()[16] ,   std::get<0>(contactInfo[i]).first->get_properties()[17] ,   std::get<0>(contactInfo[i]).first->get_properties()[18]};
        Point<dim> omegaj = {std::get<0>(contactInfo[i]).second->get_properties()[16] ,   std::get<0>(contactInfo[i]).second->get_properties()[17] ,   std::get<0>(contactInfo[i]).second->get_properties()[18]};

        Point<dim> omegaij = {0.0, 0.0, 0.0};
        double omegaNorm = (omegai - omegaj).norm();
        if(omegaNorm != 0)
        {omegaij = (omegai - omegaj) / omegaNorm ;}
        Point<dim> torquer;
       torquer = -1.0 * DEMparam.physicalProperties.murp * rEff * normalForce.norm() * omegaij;
       std::get<0>(contactInfo[i]).first->get_properties()[21] = torqueTi[0] + torquer[0];
       std::get<0>(contactInfo[i]).first->get_properties()[22] = torqueTi[1] + torquer[1];
       std::get<0>(contactInfo[i]).first->get_properties()[23] = torqueTi[2] + torquer[2];
       std::get<0>(contactInfo[i]).second->get_properties()[21] = torqueTj[0] + torquer[0];
       std::get<0>(contactInfo[i]).second->get_properties()[22] = torqueTj[1] + torquer[1];
       std::get<0>(contactInfo[i]).second->get_properties()[23] = torqueTj[2] + torquer[2];
     */


      }
    }
}


template <int dim, int spacedim>
int
ContactForce<dim, spacedim>::sgn(float a)
{
  int b = 0;
  if (a > 0)
    {
      b = 1;
    }
  else if (a < 0)
    {
      b = -1;
    }
  else if (a == 0)
    {
      b = 0;
    }
  return b;
}

template class ContactForce<3, 3>;
