//
// Created by lucka on 2021-05-21.
//

#include <core/ib_stencils.h>



template<int dim>
unsigned int
IBStencils<dim>::nb_points(unsigned int order)
{
    unsigned int nb_points=order;
    if(order>4)
        nb_points=1;
    return nb_points;
}

template<int dim>
std::vector<double>
IBStencils<dim>::coefficients(unsigned int order)
{
    if (order==1) {
        std::vector<double> coef(2);
        coef[0] = 9;
        coef[1] = -8;

        return coef;
    }
    if (order==2) {
        std::vector<double> coef(3);
        coef[0]=153;;
        coef[1]=-288;
        coef[2]=136;


        return coef;
    }
    if (order==3) {
        std::vector<double> coef(4);
        coef[0]=2925;
        coef[1]=-8424;
        coef[2]=8100;
        coef[3]=-2600;

        return coef;
    }
    if (order==4) {
        std::vector<double> coef(5);
        coef[0]=58905;
        coef[1]=-228480;
        coef[2]=332640;
        coef[3]=-215424;
        coef[4]=52360;

        return coef;
    }
    if (order>4) {
        std::vector<double> coef(1);
        coef[0] = 1;


        return coef;
    }

    std::vector<double> coef(2);
    coef[0] = 9;
    coef[1] = -8;

    return coef;
}

template<int dim>
std::tuple<Point<dim>,std::vector<Point<dim>>>
IBStencils<dim>::points(unsigned int order,IBParticle<dim> p,Point<dim> dof_point)
{
    if (order==1) {
        Tensor<1, dim, double> vect_ib = (dof_point - p.position -
                                          p.radius * (dof_point - p.position) / (dof_point - p.position).norm());




        Point<dim, double> interpolation_point_1(dof_point + vect_ib * 1 / 8);

        std::vector<Point < dim>>interpolation_points(1);
        interpolation_points[0]=interpolation_point_1;


        return {dof_point,interpolation_points};

    }
    if (order==2) {
        Tensor<1, dim, double> vect_ib =(dof_point - p.position -p.radius *(dof_point-p.position)/(dof_point-p.position).norm());


        Point<dim, double> interpolation_point_1(dof_point+vect_ib* 1./16.);

        Point<dim, double> interpolation_point_2(dof_point+vect_ib* 1./8.);

        std::vector<Point<dim>> interpolation_points(2);
        interpolation_points[0]=interpolation_point_1;
        interpolation_points[1]=interpolation_point_2;

        return {dof_point,interpolation_points};
    }
    if (order==3) {
        Tensor<1, dim, double> vect_ib =(dof_point - p.position -p.radius *(dof_point-p.position)/(dof_point-p.position).norm());

        Point<dim, double> interpolation_point_1(dof_point+vect_ib* 1./24.);

        Point<dim, double> interpolation_point_2(dof_point+vect_ib* 1/12.);

        Point<dim, double> interpolation_point_3(dof_point+vect_ib* 1./8.);

        std::vector<Point<dim>> interpolation_points(3);
        interpolation_points[0]=interpolation_point_1;
        interpolation_points[1]=interpolation_point_2;
        interpolation_points[2]=interpolation_point_3;

        return {dof_point,interpolation_points};
    }
    if (order==4) {
        Tensor<1, dim, double> vect_ib =(dof_point - p.position -p.radius *(dof_point-p.position)/(dof_point-p.position).norm());

        Point<dim, double> interpolation_point_1(dof_point+vect_ib* 1./32.);

        Point<dim, double> interpolation_point_2(dof_point+vect_ib* 1/16.);

        Point<dim, double> interpolation_point_3(dof_point+vect_ib* 3./32.);

        Point<dim, double> interpolation_point_4(dof_point+vect_ib* 1./8.);

        std::vector<Point<dim>> interpolation_points(4);
        interpolation_points[0]=interpolation_point_1;
        interpolation_points[1]=interpolation_point_2;
        interpolation_points[2]=interpolation_point_3;
        interpolation_points[3]=interpolation_point_4;

        return {dof_point,interpolation_points};
    }
    if (order>4) {
        Tensor<1, dim, double> vect_ib =(dof_point - p.position -p.radius *(dof_point-p.position)/(dof_point-p.position).norm());

        Point<dim, double> ib_point(dof_point - vect_ib);

        Point<dim, double> interpolation_point_1(dof_point+vect_ib* 1./8.);

        std::vector<Point<dim>> interpolation_points(1);
        interpolation_points[0]=interpolation_point_1;

        return {ib_point,interpolation_points};
    }

    Tensor<1, dim, double> vect_ib = (dof_point - p.position -
                                      p.radius * (dof_point - p.position) / (dof_point - p.position).norm());


    Point<dim, double> ib_point(dof_point - vect_ib);

    Point<dim, double> interpolation_point_1(dof_point + vect_ib * 1 / 8);

    std::vector<Point < dim>>interpolation_points(1);
    interpolation_points[0]=interpolation_point_1;

    return {ib_point,interpolation_points};
}


template<int dim>
double
IBStencils<dim>::vitesse_ib(IBParticle<dim> p,Point<dim> dof_point,unsigned int component) {

    double v_ib=0;
    if (dim == 2) {
        if (component == 0) {
            v_ib = -p.omega[2] *
                 p.radius *
                 ((dof_point -
                  p.position) /
                 (dof_point -
                  p.position)
                         .norm())[1] +
                    p.velocity[0];
        }
        if (component == 1) {
            v_ib = p.omega[2] *
                 p.radius *
                 ((dof_point -
                   p.position) /
                  (dof_point -
                   p.position)
                          .norm())[0] +
                 p.velocity[1];

        }
    }
    if (dim == 3) {
        if (component == 0) {
            v_ib = p.omega[1] *
                 ((dof_point -
                   p.position) /
                  (dof_point -
                   p.position)
                          .norm())[2] *
                 p.radius -
                 p.omega[2] *
                 ((dof_point -
                   p.position) /
                  (dof_point -
                   p.position)
                          .norm())[1] *
                 p.radius +
                 p.velocity[0];

        }
        if (component == 1) {
            v_ib = p.omega[2] *
                 ((dof_point -
                   p.position) /
                  (dof_point -
                   p.position)
                          .norm())[0] *
                 p.radius -
                 p.omega[0] *
                 ((dof_point -
                   p.position) /
                  (dof_point -
                   p.position)
                          .norm())[2] *
                 p.radius +
                 p.velocity[1];

        }
        if (component == 2) {
            v_ib=p.omega[0] *
                ((dof_point -
                  p.position) /
                 (dof_point -
                  p.position)
                         .norm())[1] *
                p.radius -
                p.omega[1] *
                ((dof_point-
                  p.position) /
                 (dof_point-
                  p.position)
                         .norm())[0] *
                p.radius +
                p.velocity[2];

        }

    }

    return v_ib;


}

template class IBStencils<2>;
template class IBStencils<3>;