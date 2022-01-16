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

#include "MikroPF.h"
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace
{
	inline int int_div(int a, int b)
	{
		return ((a + b - 1) / b);
	}
}

struct ParticleChunk
{
	std::vector<MikroPF::ParticleState>* particles;
	std::vector<MikroPF::Real>* weights;
	int begin, end;
	MikroPF::Real weight_sum;
	MikroPF::Real weight_sum_sq;
	bool done;
};

class MikroPF::ExecutionEngine
{
public:
	enum class Command
	{
		idle,
		run_prior,
		run_prediction_likelihood,
		normalize_weights,
		terminate
	};

	ExecutionEngine(int num_threads, int num_particles,
		int state_dim, Real effective_sample_size_th);

	~ExecutionEngine();

	bool init() noexcept;
	void execute(Command cmd);

	MikroPF::PriorFn m_prior_fn = nullptr;
	MikroPF::PredictionFn m_prediction_fn = nullptr;
	MikroPF::LikelihoodFn m_likelihood_fn = nullptr;
	MikroPF::ResamplingFn m_resampling_fn = nullptr;
	const Real* m_measurement = nullptr;
	void* m_optional_data = nullptr;
	std::vector<MikroPF::ParticleState> m_particles;
	std::vector<Real> m_state;
	std::vector<Real> m_weights;
	std::vector<std::thread> m_threads;
	std::vector<ParticleChunk> m_chunks;
	std::mutex m_mutex;
	std::condition_variable m_to_workers_cv;
	std::condition_variable m_from_workers_cv;
	int m_finished_counter = 0;
	Command m_command = Command::idle;
	const int m_num_threads;
	const int m_num_particles;
	const int m_state_dim;
	const Real m_effective_sample_size_th;

	void thread_main(ParticleChunk* chunk);
	void run_prior(ParticleChunk* chunk);
	void run_prediction_likelihood(ParticleChunk* chunk);
	void normalize_weights(ParticleChunk* chunk);
	void resample_from_indices(const std::vector<int>& indices);
};

MikroPF::ExecutionEngine::ExecutionEngine(int num_threads, int num_particles,
	int state_dim, Real effective_sample_size_th) :
	m_num_threads(num_threads), m_num_particles(num_particles), m_state_dim(state_dim),
	m_effective_sample_size_th(effective_sample_size_th)
{
}

MikroPF::ExecutionEngine::~ExecutionEngine()
{
	if (!m_threads.empty())
	{
		m_command = Command::terminate;
		m_to_workers_cv.notify_all();
		for (auto& th : m_threads)
		{
			if (th.joinable())
			{
				th.join();
			}
		}
	}
}

bool MikroPF::ExecutionEngine::init() noexcept
{
	try
	{
		m_weights.resize(m_num_particles);
		m_particles.resize(m_num_particles);
		m_state.resize(m_num_particles * m_state_dim);
		for (int i = 0; i < m_num_particles; ++i)
		{
			m_particles[i].m_state = &m_state;
			m_particles[i].m_begin = i * m_state_dim;
			m_particles[i].m_end = (i + 1) * m_state_dim;
		}
		if (m_num_threads > 1)
		{
			m_chunks.resize(m_num_threads);
			m_threads.reserve(m_num_threads);
			const int particles_per_thread = int_div(m_num_particles, m_num_threads);
			for (int i = 0; i < m_num_threads; ++i)
			{
				m_chunks[i].begin = i * particles_per_thread;
				m_chunks[i].end = std::min(m_chunks[i].begin + particles_per_thread, m_num_particles);
				m_chunks[i].particles = &m_particles;
				m_chunks[i].weights = &m_weights;
				m_chunks[i].done = false;
				m_threads.push_back(std::thread(&MikroPF::ExecutionEngine::thread_main, this, &m_chunks[i]));
			}
		}
	}
	catch (...)
	{
		return false;
	}
	return true;
}

void MikroPF::ExecutionEngine::execute(Command cmd)
{
	assert(cmd != Command::terminate);
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		m_command = cmd;
		m_to_workers_cv.notify_all();
	}

	std::unique_lock<std::mutex> lock(m_mutex);
	m_from_workers_cv.wait(lock, [&]() { return m_num_threads == m_finished_counter; });
	m_finished_counter = 0;
	m_command = Command::idle;
	for (auto& chunk : m_chunks)
	{
		chunk.done = false;
	}
}

void MikroPF::ExecutionEngine::thread_main(ParticleChunk* chunk)
{
	while (true)
	{
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_to_workers_cv.wait(lock, [&]() { return m_command != Command::idle && !chunk->done; });
		}

		if (m_command == Command::terminate)
		{
			break;
		}

		switch (m_command)
		{
		case Command::run_prior:
			run_prior(chunk);
			break;
		case Command::run_prediction_likelihood:
			run_prediction_likelihood(chunk);
			break;
		case Command::normalize_weights:
			normalize_weights(chunk);
			break;
		}

		chunk->done = true;

		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_finished_counter++;
			m_from_workers_cv.notify_one();
		}
	}
}

void MikroPF::ExecutionEngine::run_prior(ParticleChunk* chunk)
{
	for (int i = chunk->begin; i < chunk->end; ++i)
	{
		m_prior_fn(&(*chunk->particles)[i]);
	}

	const Real w = Real(1) / Real(m_num_particles);
	for (int i = chunk->begin; i < chunk->end; ++i)
	{
		(*chunk->weights)[i] = w;
	}
}

void MikroPF::ExecutionEngine::run_prediction_likelihood(ParticleChunk* chunk)
{
	chunk->weight_sum = Real(0);
	for (int i = chunk->begin; i < chunk->end; ++i)
	{
		m_prediction_fn(&(*chunk->particles)[i], m_optional_data);
		Real likelihood = Real(1);
		if (m_measurement)
		{
			m_likelihood_fn(&(*chunk->particles)[i], m_measurement, &likelihood, m_optional_data);
		}
		const Real new_weight = (*chunk->weights)[i] * likelihood;
		assert(new_weight >= 0);
		(*chunk->weights)[i] = new_weight;
		chunk->weight_sum += new_weight;
	}
}

void MikroPF::ExecutionEngine::normalize_weights(ParticleChunk* chunk)
{
	chunk->weight_sum_sq = Real(0);
	for (int i = chunk->begin; i < chunk->end; ++i)
	{
		const Real new_w = (*chunk->weights)[i] / chunk->weight_sum;
		(*chunk->weights)[i] = new_w;
		chunk->weight_sum_sq += (new_w * new_w);
	}
}

void MikroPF::ExecutionEngine::resample_from_indices(const std::vector<int>& indices)
{
	assert(indices.size() == m_num_particles);
	assert(std::is_sorted(indices.begin(), indices.end()));
	for (int i = 0; i < m_num_particles; ++i)
	{
		const int new_idx = indices[i];
		if (i == new_idx)
		{
			continue;
		}
		const MikroPF::ParticleState& p = m_particles[new_idx];
		for (int j = 0; j < m_state_dim; ++j)
		{
			m_state[i * m_state_dim + j] = p.GetAt(j);
		}
	}

	const Real w = Real(1) / Real(m_num_particles);
	for (int i = 0; i < m_num_particles; ++i)
	{
		m_weights[i] = w;
	}
}

// ==================================================================================
namespace
{
	using Real = MikroPF::Real;

	void update_multi_threaded(MikroPF::ExecutionEngine* e, const MikroPF::Real* measurement, void* optional_data)
	{
		e->m_measurement = measurement;
		e->m_optional_data = optional_data;
		e->execute(MikroPF::ExecutionEngine::Command::run_prediction_likelihood);
		e->m_measurement = nullptr;
		e->m_optional_data = nullptr;

		Real total_weight{};
		for (const auto& chunk : e->m_chunks)
		{
			total_weight += chunk.weight_sum;
		}
		for (auto& chunk : e->m_chunks)
		{
			chunk.weight_sum = total_weight;
		}
		e->execute(MikroPF::ExecutionEngine::Command::normalize_weights);

		Real total_weight_sq{};
		for (const auto& chunk : e->m_chunks)
		{
			total_weight_sq += chunk.weight_sum_sq;
		}
		const Real effective_sample_size = (Real(1) / total_weight_sq) / Real(e->m_num_particles);
		if (effective_sample_size < e->m_effective_sample_size_th)
		{
			std::vector<int> indices(e->m_num_particles);
			e->m_resampling_fn(e->m_weights, &indices);
			e->resample_from_indices(indices);
		}
	}

	void update_single_threaded(MikroPF::ExecutionEngine* e, const MikroPF::Real* measurement, void* optional_data)
	{
		Real weight_sum{};
		for (int i = 0; i < e->m_num_particles; ++i)
		{
			e->m_prediction_fn(&e->m_particles[i], optional_data);
			Real likelihood = Real(1);
			if (measurement)
			{
				e->m_likelihood_fn(&e->m_particles[i], measurement, &likelihood, optional_data);
			}
			Real new_weight = e->m_weights[i] * likelihood;
			assert(new_weight >= 0);
			e->m_weights[i] = new_weight;
			weight_sum += new_weight;
		}

		Real total_weight_sq{};
		for (int i = 0; i < e->m_num_particles; ++i)
		{
			const Real new_w = e->m_weights[i] / weight_sum;
			e->m_weights[i] = new_w;
			total_weight_sq += (new_w * new_w);
		}

		const Real effective_sample_size = (Real(1) / total_weight_sq) / Real(e->m_num_particles);
		if (effective_sample_size < e->m_effective_sample_size_th)
		{
			std::vector<int> indices(e->m_num_particles);
			e->m_resampling_fn(e->m_weights, &indices);
			e->resample_from_indices(indices);
		}
	}

	int compute_num_threads(int requested_threads)
	{
		int num_threads = requested_threads == 0 ? 1 : requested_threads;
		if (num_threads < 0)
		{
			int hw_threads = std::thread::hardware_concurrency();
			num_threads = std::max(1, hw_threads);
		}
		return num_threads;
	}
}

// ==================================================================================
MikroPF::MikroPF() = default;
MikroPF::~MikroPF() = default;

bool MikroPF::init(
	int state_dim,
	int num_particles,
	PriorFn prior_fn,
	PredictionFn prediction_fn,
	LikelihoodFn likelihood_fn,
	ResamplingFn resampling_fn,
	Real effective_sample_size_th,
	int requested_num_threads)
{
	if (state_dim < 1
		|| num_particles < 1
		|| !prior_fn
		|| !prediction_fn
		|| !likelihood_fn
		|| !resampling_fn
		|| effective_sample_size_th < 0
		|| effective_sample_size_th > 1)
	{
		return false;
	}

	const int num_threads = compute_num_threads(requested_num_threads);
	m_engine = std::make_unique<ExecutionEngine>(num_threads, num_particles, state_dim, effective_sample_size_th);
	m_engine->m_prior_fn = prior_fn;
	m_engine->m_prediction_fn = prediction_fn;
	m_engine->m_likelihood_fn = likelihood_fn;
	m_engine->m_resampling_fn = resampling_fn;
	if (!m_engine->init())
	{
		return false;
	}
	m_update_cb = num_threads == 1 ? update_single_threaded : update_multi_threaded;
	reset();
	return true;
}

void MikroPF::update(const Real * measurement, void* optional_data)
{
	m_update_cb(m_engine.get(), measurement, optional_data);
}

void MikroPF::apply(std::function<void(ParticleState* particle_state, Real* particle_weight)> callback)
{
	for (int i = 0; i < m_engine->m_num_particles; ++i)
	{
		callback(&m_engine->m_particles[i], &m_engine->m_weights[i]);
	}
}

void MikroPF::reset()
{
	if (m_engine->m_num_threads == 1)
	{
		const Real w = Real(1) / Real(m_engine->m_num_particles);
		for (int i = 0; i < m_engine->m_num_particles; ++i)
		{
			m_engine->m_weights[i] = w;
			m_engine->m_prior_fn(&m_engine->m_particles[i]);
		}
	}
	else
	{
		m_engine->execute(ExecutionEngine::Command::run_prior);
	}
}

void MikroPF::systematic_resampling(const std::vector<Real>&weights, std::vector<int>*out_indices)
{
	std::vector<Real> cumulative_sum = weights;
	const int N = cumulative_sum.size();
	for (int i = 1; i < N; ++i)
	{
		cumulative_sum[i] += cumulative_sum[i - 1];
	}
	cumulative_sum.back() = Real(1);

	Real random_offset{};
	{
		static std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
		std::uniform_real_distribution<Real> dist(0, 1);
		random_offset = dist(generator);
	}
	const Real N_inv = Real(1) / N;
	Real sample_pos = random_offset * N_inv;
	for (int i = 0, j = 0; i < N;)
	{
		if (sample_pos < cumulative_sum[j])
		{
			(*out_indices)[i] = j;
			++i;
			sample_pos = (i + random_offset) * N_inv;
		}
		else
		{
			++j;
		}
	}
}
