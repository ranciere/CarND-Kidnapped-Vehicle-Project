/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100; // TODO: Set the number of particles
  particles.clear();
  particles.resize(num_particles);
  // Random
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
  {
    particles[i] = Particle(dist_x(gen), dist_y(gen), dist_theta(gen));
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  for (auto &p : particles)
  {
    p.predict(delta_t, std_pos, velocity, yaw_rate);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations_in_car_coordinate,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  static const double EPSILON = std::numeric_limits<double>::min();
  for (auto &p : particles)
  {
    // Transform observations to map coordinate system
    std::vector<LandmarkObs> observations = p.convert_observations_to_map(observations_in_car_coordinate);

    // Match landmarks to observation and update particle weight based on multivariate gaussian
    double particle_weight = 1;

    for (const auto &obs : observations)
    {
      auto best_landmark = std::min_element(
          map_landmarks.landmark_list.begin(),
          map_landmarks.landmark_list.end(),
          [&obs, &p, sensor_range](const auto &a, const auto &b) {
            auto aval = dist(p.x, p.y, a.x_f, a.y_f) < sensor_range ? dist(a.x_f, a.y_f, obs.x, obs.y) : std::numeric_limits<double>::max();
            auto bval = dist(p.x, p.y, b.x_f, b.y_f) < sensor_range ? dist(b.x_f, b.y_f, obs.x, obs.y) : std::numeric_limits<double>::max();
            return aval < bval;
          });

      // Calculate multivariate gaussian for observation and update total weight for particle
      auto weight = multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, best_landmark->x_f, best_landmark->y_f);
      weight = std::max(EPSILON, weight);
      //
      particle_weight *= weight;
    }
    p.weight = std::max(particle_weight, EPSILON);
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<double> weights;
  weights.reserve(particles.size());

  for (const auto &p : particles)
  {
    weights.push_back(p.weight);
  }

  std::vector<Particle> new_particles;
  new_particles.reserve(particles.size());

  std::default_random_engine gen;
  std::discrete_distribution<> dist_index(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; ++i)
  {
    int index = dist_index(gen);
    new_particles.push_back(particles[index]);
  }
  std::swap(particles, new_particles);
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}