#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <vector>
#include "map.h"
#include "helper_functions.h"

class Particle
{
  static int id_cnt;
 public:
  Particle() = default;
  Particle(double _x, double _y, double _theta) : id(id_cnt++), x(_x), y(_y), theta(_theta), weight(1.0) {}
  
  void predict(double delta_t, double std_pos[], double velocity, double yaw_rate);

  std::vector<LandmarkObs> convert_observations_to_map(const std::vector<LandmarkObs>& obs) const;
  double distance(const Map::single_landmark_s& landmark) const;

  // void update_weight();

  int id;
  double x;
  double y;
  double theta;
  double weight;
  std::vector<int> associations;
  std::vector<double> sense_x;
  std::vector<double> sense_y;
};

#endif // PARTICLE_H_