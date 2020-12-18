#include <cmath>
#include <random>
#include <algorithm>

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

void Particle::updateWeight(double sensor_range, const double std_landmark[], const std::vector<LandmarkObs> &observations_in_car_coordinate, const Map &map_landmarks)
{
  static const double EPSILON = std::numeric_limits<double>::min();
 
  // Transform observations to map coordinate system
  std::vector<LandmarkObs> observations = convert_observations_to_map(observations_in_car_coordinate);

  // Match landmarks to observation and update particle weight based on multivariate gaussian
  double particle_weight = 1;
  // Iterate over observations
  for (const auto &obs : observations)
  {
    // Find minimum element (compare: distance of landmark and observation)
    auto best_landmark = std::min_element(
        map_landmarks.landmark_list.begin(),
        map_landmarks.landmark_list.end(),
        [this, &obs, sensor_range](const auto &a, const auto &b) {
          auto aval = dist(this->x, this->y, a.x_f, a.y_f) < sensor_range ? dist(a.x_f, a.y_f, obs.x, obs.y) : std::numeric_limits<double>::max();
          auto bval = dist(this->x, this->y, b.x_f, b.y_f) < sensor_range ? dist(b.x_f, b.y_f, obs.x, obs.y) : std::numeric_limits<double>::max();
          return aval < bval;
        });

    // Calculate multivariate gaussian for observation and update total weight for particle
    auto weight_ = multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, best_landmark->x_f, best_landmark->y_f);
    weight_ = std::max(EPSILON, weight_);
    // Product
    particle_weight *= weight_;
  }
  // Store the weight
  weight = std::max(particle_weight, EPSILON);
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
