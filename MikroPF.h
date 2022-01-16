// -----------------------------------------------------------------------------
// Copyright (c) 2022 Mohamed Aladem
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// -----------------------------------------------------------------------------

#pragma once

#include <cassert>
#include <vector>
#include <limits>
#include <functional>
#include <memory>

class MikroPF
{
public:
	using Real = double; // Change the floating point type used by the filter.

	// Your callbacks will be passed references to ParticleState objects.
	// You can then use them to read/write the state of particles.
	struct ParticleState
	{
	public:
		// idx must be in range [0, state_dim).
		inline Real GetAt(const int idx)const;
		inline ParticleState& SetAt(const int idx, Real val);

	private:
		friend class MikroPF;
		std::vector<Real>* m_state;
		int m_begin, m_end;
	};

	// The prior function will be used to initialize the state of each particle. Might be called from multiple threads.
	using PriorFn = void(*)(ParticleState* particle_state);

	// Used to update the state. Might be called from multiple threads.
	using PredictionFn = void(*)(ParticleState* particle_state, void* optional_data);

	// Compute likelihood p(z | sample), i.e. p(measurement | particle_state). The result out_likelihood is a single
	// number between [0,1]. Might be called from multiple threads.
	using LikelihoodFn = void(*)(const ParticleState* particle_state, const Real* measurement, Real* out_likelihood, void* optional_data);

	// Resample and report the result in out_indices. out_indices will be pre-allocated to the same size as the input
	// weights. Each element of out_indices specifies which particle to use for that position in the resample. For example, 
	// assuming we have 3 particles and out_indices is {0, 0, 2} this means copy the current particle_0 for the first 2 particles
	// of the resample, particle_1 will be dropped and current particle_2 will be copied as is.
	using ResamplingFn = void(*)(const std::vector<Real>& weights, std::vector<int>* out_indices);

	MikroPF();
	~MikroPF();

	// Initializes the filter. Returns true on success.
	bool init(
		int state_dim,
		int num_particles,
		PriorFn prior_fn,
		PredictionFn prediction_fn,
		LikelihoodFn likelihood_fn,
		ResamplingFn resampling_fn,
		Real effective_sample_size_th, // Perform resampling if effective sample size drops below this threshold. Range [0, 1].
		int num_threads = -1 // A negative number indicates auto-detect the number of threads. 
	);

	// Update the filter. The new measurement will be incorporated if available. The measurement here will be
	// passed as-is to the LikelihoodFn, the filter itself imposes no requirements on its dimensions.
	void update(const Real* measurement = nullptr, void* optional_data = nullptr);

	// Apply a function on each particle and its weight. This will be called from the main thread.
	void apply(std::function<void(ParticleState* particle_state, Real* particle_weight)> callback);

	// Reinitializes the particles using the prior_fn and resets their weights.
	void reset();

	// ------------------
	// Resampling methods
	// ------------------
	static void systematic_resampling(const std::vector<Real>& weights, std::vector<int>* out_indices);

	class ExecutionEngine; // internal - don't worry about it.

private:
	std::unique_ptr<ExecutionEngine> m_engine;
	void(*m_update_cb)(ExecutionEngine* e, const Real* measurement, void* optional_data) = nullptr;
};

// ==================================================================================
inline MikroPF::ParticleState& MikroPF::ParticleState::SetAt(const int idx, Real val)
{
	const int i = m_begin + idx;
	assert(i >= m_begin && i < m_end);
	if (i >= m_begin && i < m_end)
	{
		(*m_state)[i] = val;
	}
	return *this;
}

inline MikroPF::Real MikroPF::ParticleState::GetAt(const int idx)const
{
	const int i = m_begin + idx;
	assert(i >= m_begin && i < m_end);
	return ((i >= m_begin && i < m_end) ? (*m_state)[i] : std::numeric_limits<Real>::infinity());
}

