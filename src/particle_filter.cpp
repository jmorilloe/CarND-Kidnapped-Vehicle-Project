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

#include "particle_filter.h"

using namespace std;

static default_random_engine re;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	weights.resize(num_particles);

	// Define normal distributions
	normal_distribution<double> error_x(x, std[0]);
	normal_distribution<double> error_y(y, std[1]);
	normal_distribution<double> error_p(theta, std[2]);

	// Init particles
	for (int i=0; i < num_particles; i++) {
		Particle particle = { i, error_x(re), error_y(re), error_p(re), 1.0 };
		particles.emplace_back(particle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> error_x(0, std_pos[0]);
	normal_distribution<double> error_y(0, std_pos[1]);
	normal_distribution<double> error_p(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {

		if (fabs(yaw_rate) > 0.00001) {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}

		particles[i].x += error_x(re);
		particles[i].y += error_y(re);
		particles[i].theta += error_p(re);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {

		double min_distance = numeric_limits<double>::max();

		for (int j = 0; j < predicted.size(); j++) {

			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance < min_distance) {
				min_distance = distance;
				observations[i].id = predicted[j].id;
			}
		}
	}
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();

	for (int i = 0; i < num_particles; i++) {

		double cur_x = particles[i].x;
		double cur_y = particles[i].y;
		double cur_theta = particles[i].theta;

		vector<LandmarkObs> predicted_landmarks;

		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			double distance = dist(cur_x, cur_y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);

			if (distance <= sensor_range) {

				LandmarkObs landmark;
				landmark.id = map_landmarks.landmark_list[j].id_i;
				landmark.x = map_landmarks.landmark_list[j].x_f;
				landmark.y = map_landmarks.landmark_list[j].y_f;
				predicted_landmarks.push_back(landmark);
			}
		}

		vector<LandmarkObs> observations_map;

		for (int j = 0; j < observations.size(); j++) {

			LandmarkObs landmark;
			landmark.x = observations[j].x * cos(cur_theta) - observations[j].y * sin(cur_theta) + cur_x;
			landmark.y = observations[j].x * sin(cur_theta) + observations[j].y * cos(cur_theta) + cur_y;
			observations_map.push_back(landmark);
		}

		dataAssociation(predicted_landmarks, observations_map);
		double weight = 1.0;

		for (int k = 0; k < predicted_landmarks.size(); k++) {

			int min_index = -1;
		
			for (int m = 0; m < observations_map.size(); m++) {

				if (predicted_landmarks[k].id == observations_map[m].id) {
		
					min_index = m;
					break;
				}
			}

			double delta_x = predicted_landmarks[k].x - observations_map[min_index].x;
			double delta_y = predicted_landmarks[k].y - observations_map[min_index].y;
			weight *= exp(-0.5 * (pow(delta_x, 2.0) * std_landmark[0] + pow(delta_y, 2.0) * std_landmark[1])) / sqrt(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
		}

		weights.push_back(weight);
		particles[i].weight = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	discrete_distribution<int> distribution_weights(weights.begin(), weights.end());
	vector<Particle> new_particles;

	for(int i = 0; i < num_particles; i++) {

		Particle particle = particles[distribution_weights(re)];
		new_particles.push_back(particle);
	}

	particles = new_particles;
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