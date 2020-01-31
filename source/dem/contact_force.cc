﻿/*
 * contactForce.cpp
 *
 *  Created on: Oct 31, 2019
 *      Author: shahab
 */


#include "dem/contact_force.h"

#include <deal.II/base/point.h>

#include <boost/math/special_functions/sign.hpp>

#include "dem/dem_iterator.h"

using namespace dealii;

template <int dim, int spacedim>
ContactForce<dim, spacedim>::ContactForce()
{}

template <int dim, int spacedim>
void
ContactForce<dim, spacedim>::linearCF(
  std::vector<std::map<int, ContactInfoStruct<dim, spacedim>>> inContactInfo,
  int                                                          Yp,
  float                                                        vp,
  float                                                        ep,
  float                                                        mup,
  float                                                        murp)
{
  typename std::map<int, ContactInfoStruct<dim, spacedim>>::iterator info_it;
  // delete particleI from the list, loop over particles here instead
  for (unsigned int i = 0; i < inContactInfo.size(); i++)
    {
      info_it = inContactInfo[i].begin();
      while (info_it != inContactInfo[i].end())
        {
          Point<dim> totalForce;
          double     mEff = (info_it->second.particleI->get_properties()[19] *
                         info_it->second.particleJ->get_properties()[19]) /
                        (info_it->second.particleI->get_properties()[19] +
                         info_it->second.particleJ->get_properties()[19]);
          double rEff =
            (info_it->second.particleI->get_properties()[2] *
             info_it->second.particleJ->get_properties()[2]) /
            (2.0 * (info_it->second.particleI->get_properties()[2] +
                    info_it->second.particleJ->get_properties()[2]));
          double yEff  = Yp / (2.0 * (1.0 - pow(vp, 2.0)));
          double kn    = 1.2024 * pow((pow(mEff, 0.5) * pow(yEff, 2.0) * rEff *
                                    abs(info_it->second.normRelVel)),
                                   0.4);
          double kt    = 1.2024 * pow((pow(mEff, 0.5) * pow(yEff, 2.0) * rEff *
                                    abs(info_it->second.tangRelVel)),
                                   0.4);
          double ethan = (-2.0 * log(ep) * sqrt(mEff * kn)) /
                         (sqrt((pow(log(ep), 2.0)) + pow(3.1415, 2.0)));
          double ethat = 0;
          if (ep == 0)
            {
              ethat = 2.0 * sqrt(2.0 / 7.0 * mEff * kt);
            }
          else
            {
              ethat = (-2.0 * log(ep) * sqrt(2.0 / 7.0 * mEff * kt)) /
                      (sqrt(pow(3.1415, 2.0) + pow(log(ep), 2.0)));
            }

          Point<dim> springNormForce =
            (kn * info_it->second.normOverlap) * info_it->second.normVec;
          Point<dim> dashpotNormForce =
            (ethan * info_it->second.normRelVel) * info_it->second.normVec;

          Point<dim> normalForce;
          normalForce = springNormForce + dashpotNormForce;

          Point<dim> springTangForce =
            (kt * info_it->second.tangOverlap) * info_it->second.tangVec;
          Point<dim> dashpotTangForce =
            (ethat * info_it->second.tangRelVel) * info_it->second.tangVec;

          Point<dim> tangForce;
          tangForce = springTangForce + dashpotTangForce;

          if (tangForce.norm() < (mup * normalForce.norm()))
            {
              totalForce = normalForce + tangForce;
            }
          else
            {
              Point<dim> coulumbTangForce =
                (mup * normalForce.norm() *
                 boost::math::sign(info_it->second.tangOverlap)) *
                info_it->second.tangVec;

              totalForce = normalForce + coulumbTangForce;
            }

          info_it->second.particleI->get_properties()[13] =
            info_it->second.particleI->get_properties()[13] - totalForce[0];
          info_it->second.particleI->get_properties()[14] =
            info_it->second.particleI->get_properties()[14] - totalForce[1];
          info_it->second.particleI->get_properties()[15] =
            info_it->second.particleI->get_properties()[15] - totalForce[2];

          info_it->second.particleJ->get_properties()[13] =
            info_it->second.particleJ->get_properties()[13] + totalForce[0];
          info_it->second.particleJ->get_properties()[14] =
            info_it->second.particleJ->get_properties()[14] + totalForce[1];
          info_it->second.particleJ->get_properties()[15] =
            info_it->second.particleJ->get_properties()[15] + totalForce[2];

          // calculation of torque
          /*
           Point<dim> torqueTi;
           torqueTi =
         ((std::get<0>(contactInfo[i]).first->get_properties()[2])/2.0) *
         cross_product_3d( std::get<2>(contactInfo[i]) , totalForce); Point<dim>
         torqueTj; torqueTj =
         ((std::get<0>(contactInfo[i]).second->get_properties()[2])/2.0) *
         cross_product_3d( std::get<2>(contactInfo[i]) , -1.0*totalForce);
          Point<dim> omegai =
         {std::get<0>(contactInfo[i]).first->get_properties()[16] ,
         std::get<0>(contactInfo[i]).first->get_properties()[17] ,
         std::get<0>(contactInfo[i]).first->get_properties()[18]}; Point<dim>
         omegaj = {std::get<0>(contactInfo[i]).second->get_properties()[16] ,
         std::get<0>(contactInfo[i]).second->get_properties()[17] ,
         std::get<0>(contactInfo[i]).second->get_properties()[18]};

          Point<dim> omegaij = {0.0, 0.0, 0.0};
          double omegaNorm = (omegai - omegaj).norm();
          if(omegaNorm != 0)
          {omegaij = (omegai - omegaj) / omegaNorm ;}
          Point<dim> torquer;
         torquer = -1.0 * murp * rEff * normalForce.norm() * omegaij;
         std::get<0>(contactInfo[i]).first->get_properties()[21] = torqueTi[0] +
         torquer[0]; std::get<0>(contactInfo[i]).first->get_properties()[22] =
         torqueTi[1] + torquer[1];
         std::get<0>(contactInfo[i]).first->get_properties()[23] = torqueTi[2] +
         torquer[2]; std::get<0>(contactInfo[i]).second->get_properties()[21] =
         torqueTj[0] + torquer[0];
         std::get<0>(contactInfo[i]).second->get_properties()[22] = torqueTj[1]
         + torquer[1]; std::get<0>(contactInfo[i]).second->get_properties()[23]
         = torqueTj[2] + torquer[2];
       */

          info_it++;
        }
    }
}

template <int dim, int spacedim>
void
ContactForce<dim, spacedim>::nonLinearCF(
  std::vector<std::map<int, ContactInfoStruct<dim, spacedim>>> inContactInfo,
  int                                                          Yp,
  float                                                        vp,
  float                                                        mup,
  float                                                        murp)
{
  typename std::map<int, ContactInfoStruct<dim, spacedim>>::iterator info_it;

  for (unsigned int i = 0; i < inContactInfo.size(); i++)
    {
      info_it = inContactInfo[i].begin();

      while (info_it != inContactInfo[i].end())
        {
          Point<dim> totalForce;
          double     mEff = (info_it->second.particleI->get_properties()[19] *
                         info_it->second.particleJ->get_properties()[19]) /
                        (info_it->second.particleI->get_properties()[19] +
                         info_it->second.particleJ->get_properties()[19]);
          double rEff =
            (info_it->second.particleI->get_properties()[2] *
             info_it->second.particleJ->get_properties()[2]) /
            (2.0 * (info_it->second.particleI->get_properties()[2] +
                    info_it->second.particleJ->get_properties()[2]));
          double yEff  = Yp / (2.0 * (1.0 - pow(vp, 2.0)));
          double gEff  = (Yp) / (4.0 * (2.0 - vp) * (1.0 + vp));
          double betha = log(vp) / sqrt(pow(log(vp), 2.0) + 9.8696);
          double sn    = 2.0 * yEff * sqrt(rEff * info_it->second.normOverlap);
          double st    = 8.0 * gEff * sqrt(rEff * info_it->second.normOverlap);
          double kn = 1.3333 * yEff * sqrt(rEff * info_it->second.normOverlap);
          double ethan = -1.8257 * betha * sqrt(sn * mEff);
          double kt    = 8.0 * gEff * sqrt(rEff * info_it->second.normOverlap);
          double ethat = -1.8257 * betha * sqrt(st * mEff);

          Point<dim> springNormForce =
            (kn * info_it->second.normOverlap) * info_it->second.normVec;
          Point<dim> dashpotNormForce =
            (ethan * info_it->second.normRelVel) * info_it->second.normVec;

          Point<dim> normalForce;
          normalForce = springNormForce + dashpotNormForce;

          Point<dim> springTangForce =
            (kt * info_it->second.tangOverlap) * info_it->second.tangVec;
          Point<dim> dashpotTangForce =
            (ethat * info_it->second.tangRelVel) * info_it->second.tangVec;

          Point<dim> tangForce;
          tangForce = springTangForce + dashpotTangForce;

          if (tangForce.norm() < (mup * normalForce.norm()))
            {
              totalForce = normalForce + tangForce;
            }
          else
            {
              Point<dim> coulumbTangForce =
                (mup * normalForce.norm() *
                 boost::math::sign(info_it->second.tangOverlap)) *
                info_it->second.tangVec;
              totalForce = normalForce + coulumbTangForce;
            }
          info_it->second.particleI->get_properties()[13] =
            info_it->second.particleI->get_properties()[13] - totalForce[0];
          info_it->second.particleI->get_properties()[14] =
            info_it->second.particleI->get_properties()[14] - totalForce[1];
          info_it->second.particleI->get_properties()[15] =
            info_it->second.particleI->get_properties()[15] - totalForce[2];

          info_it->second.particleJ->get_properties()[13] =
            info_it->second.particleJ->get_properties()[13] + totalForce[0];
          info_it->second.particleJ->get_properties()[14] =
            info_it->second.particleJ->get_properties()[14] + totalForce[1];
          info_it->second.particleJ->get_properties()[15] =
            info_it->second.particleJ->get_properties()[15] + totalForce[2];

          // calculation of torque
          /*
           Point<dim> torqueTi;
           torqueTi =
         ((std::get<0>(contactInfo[i]).first->get_properties()[2])/2.0) *
         cross_product_3d( std::get<2>(contactInfo[i]) , totalForce); Point<dim>
         torqueTj; torqueTj =
         ((std::get<0>(contactInfo[i]).second->get_properties()[2])/2.0) *
         cross_product_3d( std::get<2>(contactInfo[i]) , -1.0*totalForce);
          Point<dim> omegai =
         {std::get<0>(contactInfo[i]).first->get_properties()[16] ,
         std::get<0>(contactInfo[i]).first->get_properties()[17] ,
         std::get<0>(contactInfo[i]).first->get_properties()[18]}; Point<dim>
         omegaj = {std::get<0>(contactInfo[i]).second->get_properties()[16] ,
         std::get<0>(contactInfo[i]).second->get_properties()[17] ,
         std::get<0>(contactInfo[i]).second->get_properties()[18]};

          Point<dim> omegaij = {0.0, 0.0, 0.0};
          double omegaNorm = (omegai - omegaj).norm();
          if(omegaNorm != 0)
          {omegaij = (omegai - omegaj) / omegaNorm ;}
          Point<dim> torquer;
         torquer = -1.0 * murp * rEff * normalForce.norm() * omegaij;
         std::get<0>(contactInfo[i]).first->get_properties()[21] = torqueTi[0] +
         torquer[0]; std::get<0>(contactInfo[i]).first->get_properties()[22] =
         torqueTi[1] + torquer[1];
         std::get<0>(contactInfo[i]).first->get_properties()[23] = torqueTi[2] +
         torquer[2]; std::get<0>(contactInfo[i]).second->get_properties()[21] =
         torqueTj[0] + torquer[0];
         std::get<0>(contactInfo[i]).second->get_properties()[22] = torqueTj[1]
         + torquer[1]; std::get<0>(contactInfo[i]).second->get_properties()[23]
         = torqueTj[2] + torquer[2];
       */
          info_it++;
        }
    }
}


template class ContactForce<3, 3>;
