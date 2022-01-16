#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>

#include "MikroPF.h"

using Real = MikroPF::Real;

void prior_fn(MikroPF::ParticleState* particle_state)
{
  // Initialize from a gaussian.
  thread_local std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
  std::normal_distribution<Real> norm(0.0, 1.0);
  particle_state->SetAt(0, norm(generator)).SetAt(1, norm(generator)).SetAt(2, norm(generator));
}

void prediction_fn(MikroPF::ParticleState* particle_state, void* /*optional_data*/)
{
  // Simple linear dynamics model with added gaussian noise.
  constexpr Real dt = 0.25;
  constexpr Real noise_sigma = 0.125;
  thread_local std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
  std::normal_distribution<Real> norm(0.0, noise_sigma);
  const Real x = particle_state->GetAt(0);
  const Real v = particle_state->GetAt(1);
  const Real a = particle_state->GetAt(2);
  Real new_a = a + norm(generator);
  Real new_v = v + a * dt + norm(generator);
  Real new_x = x + v * dt + 0.5 * a * dt * dt + norm(generator);
  particle_state->SetAt(0, new_x).SetAt(1, new_v).SetAt(2, new_a);
}

void likelihood_fn(const MikroPF::ParticleState* particle_state, const Real* measurement,
  Real* out_likelihood, void* /*optional_data*/)
{
  constexpr Real noise_sigma = 1.0;
  const Real error = particle_state->GetAt(0) - *measurement;
  // Map the error to a probability using an RBF kernel.
  *out_likelihood = std::exp(-(error * error) / (2.0f * noise_sigma * noise_sigma));
}

int main()
{
  // Tracking a sinusoidal. This example is adapted from: https://github.com/johnhw/pfilter/blob/master/examples/timeseries.ipynb
  constexpr int N = 100;
  std::vector<int> x(N);
  std::vector<Real> y_true(N);
  std::vector<Real> y_noisy(N);
  std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
  // Fill the data.
  {
    std::normal_distribution<Real> norm(0.0, 0.5);
    for (int i = 0; i < N; ++i)
    {
      x[i] = i;
      y_true[i] = std::cos(0.25 * x[i]) + x[i] * 0.05;
      y_noisy[i] = y_true[i] + norm(generator);
    }
  }

  // Create and initialize our filter.
  MikroPF pf;
  bool result = pf.init(
    /*state_dim*/ 3, // state is [x, dx, ddx]
    /*num_particles*/ 2000,
    prior_fn,
    prediction_fn,
    likelihood_fn,
    MikroPF::systematic_resampling,
    /*effective_sample_size_th*/ 0.25,
    /*num_threads*/ -1);
  if (!result)
  {
    std::cout << "Particle filter initialization failed." << std::endl;
    return -1;
  }

  std::uniform_real_distribution<Real> dist(0, 1);
  std::vector<Real> estimated_positions(N);
  for (int i = 0; i < N; ++i)
  {
    // Note that our measurement here is a single floating point value but that need not be the case. The measurement could be an array of values for example.
    Real* measurement = dist(generator) > 0.25 ? &y_noisy[i] : nullptr; // Randomly drop some measurements to test the filter result with missing values.
    pf.update(measurement);
    // Now we apply a function to compute our mean estimated position value.
    std::vector<MikroPF::Real> mean_state(3, 0.0);
    pf.apply([&mean_state](MikroPF::ParticleState* particle_state, Real* particle_weight) {
      mean_state[0] += (*particle_weight * particle_state->GetAt(0));
      mean_state[1] += (*particle_weight * particle_state->GetAt(1));
      mean_state[2] += (*particle_weight * particle_state->GetAt(2));
      });
    estimated_positions[i] = mean_state[0];
  }

  // Write into a csv file for plotting.
  {
    std::ofstream file("result.csv");
    file << "x,y_true,y_noisy,pf" << std::endl;
    for (int i = 0; i < N; ++i)
    {
      file << x[i] << "," << y_true[i] << "," << y_noisy[i] << "," << estimated_positions[i] << std::endl;
    }
  }
}