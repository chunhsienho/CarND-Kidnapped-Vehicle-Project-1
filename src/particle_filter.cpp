/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"
using namespace std;
const double LARGE_DISTANCE = 1e10;

double calc_dist(double dx, double dy) {
  return sqrt(dx*dx + dy*dy);
}


using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  default_random_engine gen;
	normal_distribution<double> N_x_init(x, std[0]);
	normal_distribution<double> N_y_init(y, std[1]);
	normal_distribution<double> N_theta_init(theta, std[2]);
  vector<Particle> initial_particles;

  for(int i=0; i<num_particles; i++){
    Particle p;
    p.id = i;
    p.x = N_x_init(gen);
    p.y = N_y_init(gen);
    p.theta = N_theta_init(gen);
    p.weight = 1.0;
    initial_particles.push_back(p);
  }

  particles = initial_particles;
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  default_random_engine gen;
	normal_distribution<double> N_x_pred(0.0f, std_pos[0]);
	normal_distribution<double> N_y_pred(0.0f, std_pos[1]);
	normal_distribution<double> N_theta_pred(0.0f, std_pos[2]);

  for(int i=0; i<particles.size(); i++) {
    Particle p = particles[i];
    double noise_x = N_x_pred(gen);
    double noise_y = N_y_pred(gen);
    double noise_theta = N_theta_pred(gen);

    double dx =  velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
    double dy =  velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
    p.x += dx + noise_x;
    p.y += dy + noise_y;
    p.theta = p.theta + yaw_rate * delta_t + noise_theta;

    particles[i] = p;
  }


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  // find closest landmark for each observation
  vector<LandmarkObs> associated_observations;
  for (vector<LandmarkObs>::iterator o_it = observations.begin() ; o_it != observations.end(); ++o_it) {
    LandmarkObs obs = *o_it;
    double min_dist = LARGE_DISTANCE;
    for (vector<LandmarkObs>::iterator p_it = predicted.begin() ; p_it != predicted.end(); ++p_it) {
      LandmarkObs pred_obs = *p_it;
      double distance = calc_dist(obs.x - pred_obs.x, obs.y - pred_obs.y);
      if (distance < min_dist) {
        min_dist = distance;
        obs.id = pred_obs.id;
      }
    }
    associated_observations.push_back(obs);
  }
  observations = associated_observations;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

  for (int i=0; i<particles.size(); i++) {

    // predict observations using map and predicted particle locations
    Particle particle = particles[i];
    vector<LandmarkObs> pred_observations;
    map<int, LandmarkObs> pred_observations_map;  // stores id prediction mapping

    for(int j=0; j<map_landmarks.landmark_list.size(); j++){
      // Tanslate landmark to car coordinate space
      // 1. translate by particle.x particle.y
      Map::single_landmark_s l = map_landmarks.landmark_list[j];
      double px = l.x_f - particle.x;
      double py = l.y_f - particle.y;
      double obs_dist = calc_dist(px, py);

      // skip landmark out of observation distance
      if(obs_dist > sensor_range){
        continue;
      }

      // 2. rotate by particle.theta
      double cos_theta = cos(-particle.theta);
      double sin_theta = sin(-particle.theta);
      LandmarkObs obs_pred;
      obs_pred.x = px * cos_theta - py * sin_theta;
      obs_pred.y = px * sin_theta + py * cos_theta;
      obs_pred.id = l.id_i;
      pred_observations.push_back(obs_pred);

      // add prediction to map
      pred_observations_map[l.id_i] = obs_pred;
    }

    // associate observations to landmarks
    dataAssociation(pred_observations, observations);

    // calculate weight using multivariate normal distrubution model
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double weight = 1.0f;
    // double norm = 1.0f / ( 2 * M_PI * sig_x * sig_y);  // normalizer
    double norm = 1.0f;

    double sig_x2 = sig_x * sig_x;
    double sig_y2 = sig_y * sig_y;
    for (vector<LandmarkObs>::iterator o_it = observations.begin() ; o_it != observations.end(); ++o_it) {
      LandmarkObs obs = *o_it;
      LandmarkObs pred_obs = pred_observations_map[obs.id];
      double dx = obs.x - pred_obs.x;
      double dy = obs.y - pred_obs.y;
      double single_prob = norm * exp(-((dx*dx)/(2*sig_x2) + (dy*dy)/(2.0*sig_y2)));
      weight *= single_prob;
    }
    particles[i].weight = weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<double> weights;
  for (vector<Particle>::iterator it = particles.begin() ; it != particles.end(); ++it) {
    Particle particle = *it;
    weights.push_back(particle.weight);
  }

  default_random_engine gen;
  discrete_distribution<> resample_index(weights.begin(), weights.end());

  vector<Particle> new_particles;
  for(int i=0; i<num_particles; i++){
    int index = resample_index(gen);
    Particle new_particle = particles[index];
    new_particles.push_back(new_particle);
  }
  particles = new_particles;
  // cout << "New particles:\n";
  // for(int i=0; i<particles.size(); i++){
  //   Particle p = particles[i];
  //   cout << "(" << p.x << "," << p.y << ") ";
  // }
  // cout << "\n";

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
