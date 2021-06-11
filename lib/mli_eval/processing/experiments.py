import os
import json
import torch
import numpy as np

import mli.metrics as metrics_utils


def get_run_stats(expdir, compute_weight_dist=True):
	alldirs = os.listdir(expdir)
	configs = []
	metrics = []
	for rundir in alldirs:
		# Sacred saves all source code
		if rundir == "_sources":
			continue
		dirpath = os.path.join(expdir, rundir)
		if not os.path.isdir(dirpath):
			continue
		config_f = os.path.join(dirpath, "config.json")
		metrics_f = os.path.join(dirpath, "metrics.json")
		init_f = os.path.join(expdir, rundir, "init.pt")
		final_f = os.path.join(expdir, rundir, "final.pt")

		# Check directory for completeness:
		valid = True
		valid = valid and os.path.isfile(config_f)
		valid = valid and os.path.isfile(metrics_f)
		valid = valid and os.path.isfile(init_f)
		valid = valid and os.path.isfile(final_f)
		if not valid:
			print("Incomplete experiment output in {}".format(dirpath))
			continue
		with open(config_f, "r") as f:
			data = json.load(f)
			data["run_id"] = os.path.split(rundir)[1]
			configs.append(data)
		with open(metrics_f, "r") as f:
			metric_data = json.load(f)
			# Get last entries only - protects against preemption
			steps = data["alpha_steps"]
			metric_data["train.interpolation.alpha"]["values"] = metric_data["train.interpolation.alpha"]["values"][
																 -steps:]
			metric_data["train.interpolation.loss"]["values"] = metric_data["train.interpolation.loss"]["values"][
																-steps:]
			metric_data["train.interpolation.acc"]["values"] = metric_data["train.interpolation.acc"]["values"][-steps:]
			metric_data["eval.interpolation.alpha"]["values"] = metric_data["eval.interpolation.alpha"]["values"][
																-steps:]
			metric_data["eval.interpolation.loss"]["values"] = metric_data["eval.interpolation.loss"]["values"][-steps:]
			metric_data["eval.interpolation.acc"]["values"] = metric_data["eval.interpolation.acc"]["values"][-steps:]
		if compute_weight_dist:
			param1 = torch.load(init_f)
			param2 = torch.load(final_f)
			dist = metrics_utils.param_dist(param1["model_state"], param2["model_state"])
			metric_data["weight_dist"] = dist
			dist = metrics_utils.param_dist(param1["model_state"], param2["model_state"], True)
			metric_data["normed_weight_dist"] = dist
		else:
			continue
		metrics.append(metric_data)

	print("Found {} runs".format(len(configs)))
	return configs, metrics


def get_run_lm_stats(expdir, compute_weight_dist=False):
	alldirs = os.listdir(expdir)
	configs = []
	metrics = []
	for rundir in alldirs:
		# Sacred saves all source code
		if rundir == "_sources":
			continue
		dirpath = os.path.join(expdir, rundir)
		if not os.path.isdir(dirpath):
			continue
		config_f = os.path.join(dirpath, "config.json")
		metrics_f = os.path.join(dirpath, "metrics.json")

		# Check directory for completeness:
		valid = True
		valid = valid and os.path.isfile(config_f)
		valid = valid and os.path.isfile(metrics_f)
		if not valid:
			print("Incomplete experiment output in {}".format(dirpath))
			continue
		with open(config_f, "r") as f:
			data = json.load(f)
			data["run_id"] = os.path.split(rundir)[1]
			configs.append(data)
		with open(metrics_f, "r") as f:
			metric_data = json.load(f)
			try:
				# Get last entries only --- protects against preemption
				steps = data["alpha_steps"]
				metric_data["train.interpolation.alpha"]["values"] = metric_data["train.interpolation.alpha"]["values"][
																	 -steps:]
				metric_data["train.interpolation.loss"]["values"] = metric_data["train.interpolation.loss"]["values"][
																	-steps:]
				# metric_data["val.interpolation.alpha"]["values"] = metric_data["val.interpolation.alpha"]["values"][
				# 													-steps:]
				metric_data["val.interpolation.loss"]["values"] = metric_data["val.interpolation.loss"]["values"][-steps:]
			except:
				pass
		metrics.append(metric_data)

	print("Found {} runs".format(len(configs)))
	return configs, metrics


def get_run_model_states(rundir):
	ret = {}
	config_f = os.path.join(rundir, "config.json")
	with open(config_f, "r") as f:
		data = json.load(f)
		data["run_id"] = os.path.split(rundir)[1]
		ret["config"] = data

	metric_f = os.path.join(rundir, "metrics.json")
	with open(metric_f, "r") as f:
		ret["metrics"] = json.load(f)

	init_f = os.path.join(rundir, "init.pt")
	with open(metric_f, "r") as f:
		ret["init_state"] = torch.load(init_f)["model_state"]

	final_f = os.path.join(rundir, "final.pt")
	with open(metric_f, "r") as f:
		ret["final_state"] = torch.load(final_f)["model_state"]

	return ret


def get_monotonicity_metrics(all_configs, all_metrics):
	summary = []
	for i in range(len(all_configs)):
		metrics = all_metrics[i]

		try:
			alphas = metrics["train.interpolation.alpha"]["values"]
			losses = metrics["train.interpolation.loss"]["values"]
			_, heights = metrics_utils.eval_monotonic(alphas, losses)
			if len(heights) > 0:
				max_bump = np.max(heights)
			else:
				max_bump = 0

			stats = {"run": all_configs[i]["run_id"], "max_bump": max_bump, "config": all_configs[i],
					 "normed_weight_dist": metrics["train.norm_wdist"]["values"][-1]}
			try:
				stats["avg_gl"] = metrics["gauss_len"]["values"][-1]
			except:
				stats["avg_gl"] = 0.

			summary.append(
				stats
			)
		except:
			pass
	return summary


def summarize_metrics(all_configs, all_metrics,
					  metric_summaries=[(np.min, "train.loss"), (np.max, "train.acc")]):
	summaries = {}
	for ms in metric_summaries:
		summaries[ms[1]] = {
			"max_id": None,
			"max_val": 0,
			"min_id": None,
			"min_val": 1e14
		}
	for i in range(len(all_configs)):
		metrics = all_metrics[i]
		for ms in metric_summaries:
			val = ms[0](metrics[ms[1]]["values"])
			summary = summaries[ms[1]]
			if summary["max_id"] is None or val > summary["max_val"]:
				summary["max_val"] = val
				summary["max_id"] = all_configs[i]["run_id"]
			if summary["min_val"] is None or val < summary["min_val"]:
				summary["min_val"] = val
				summary["min_id"] = all_configs[i]["run_id"]
	return summaries


def summarize_lm_metrics(all_configs, all_metrics,
						 metric_summaries=[(np.min, "train.ppl")]):
	summaries = {}
	for ms in metric_summaries:
		summaries[ms[1]] = {
			"max_id": None,
			"max_val": 0,
			"min_id": None,
			"min_val": 1e14
		}
	for i in range(len(all_configs)):
		metrics = all_metrics[i]
		for ms in metric_summaries:
			try:
				val = ms[0](metrics[ms[1]]["values"])
				summary = summaries[ms[1]]
				if summary["max_id"] is None or val > summary["max_val"]:
					summary["max_val"] = val
					summary["max_id"] = all_configs[i]["run_id"]
				if summary["min_val"] is None or val < summary["min_val"]:
					summary["min_val"] = val
					summary["min_id"] = all_configs[i]["run_id"]
			except:
				pass
	return summaries
