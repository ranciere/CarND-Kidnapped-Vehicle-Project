#include <cmath>
#include <random>
#include "particle.h"

int Particle::id_cnt = 0;

void Particle::predict(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
  std::default_random_engine gen;
  auto delta_x = 0.0;
  auto delta_y = 0.0;
  auto delta_theta = 0.0;

  // Straight
  if (fabs(yaw_rate) <= 1E-6)
  {
    delta_theta = 0;
    delta_x = velocity * delta_t * sin(theta);
    delta_y = velocity * delta_t * cos(theta);
  }
  // Non-straight
  else
  {
    delta_theta = yaw_rate * delta_t;
    delta_x = velocity / yaw_rate * (sin(theta + delta_theta) - sin(theta));
    delta_y = velocity / yaw_rate * (cos(theta) - cos(theta + delta_theta));
  }
  // Distribution of noises
  std::normal_distribution<double> dist_x(x + delta_x, std_pos[0]);
  std::normal_distribution<double> dist_y(y + delta_y, std_pos[1]);
  std::normal_distribution<double> dist_theta(theta + delta_theta, std_pos[2]);
  // Add noise
  x = dist_x(gen);
  y = dist_y(gen);
  theta = dist_theta(gen);
}

std::vector<LandmarkObs> Particle::convert_observations_to_map(const std::vector<LandmarkObs> &obs) const
{
  std::vector<LandmarkObs> result;
  result.reserve(obs.size());

  for (const auto &o: obs)
  {
      LandmarkObs p_obs;
      p_obs.x = (o.x * cos(theta)) - (o.y * sin(theta)) + x;
      p_obs.y = (o.x * sin(theta)) + (o.y * cos(theta)) + y;
      result.push_back(p_obs);
  }
  return result;
}

double Particle::distance(const Map::single_landmark_s& landmark) const
{
  return dist(x, y, landmark.x_f, landmark.y_f);
}
