# helper utils for variables extraction and insertion into models


def extract_params_and_shapes_and_keys(model):
	shapes_list = []
	vals_list = []
	keys_list = []

	for i, state_now in enumerate(model.state_dict()):
		vals_now = model.state_dict()[state_now].detach().cpu().numpy()
		shape_now = model.state_dict()[state_now].detach().cpu().numpy().shape
		key_now = state_now

		shapes_list.append(shape_now)
		vals_list.append(vals_now.reshape([-1]))
		keys_list.append(key_now)

	return keys_list, shapes_list, np.concatenate(vals_list, axis=0)


def reform_params(keys_list, shapes_list, flat_params):
	index_now = 0
	reshaped_params_list = []

	new_dict = dict()

	for i, shape_now in enumerate(shapes_list):
		size_now = np.prod(shape_now)
		key_now = keys_list[i]

		cut_params_now = flat_params[index_now:index_now + size_now]
		reshaped_params_list.append(cut_params_now.reshape(shape_now))
		index_now = index_now + size_now

		new_dict[key_now] = torch.Tensor(cut_params_now.reshape(shape_now))

	return new_dict


# an example of usage
keys_list, shapes_list, init_params_vector = extract_params_and_shapes_and_keys(model)  # final vector
print("Number of params at init = " + str(len(init_params_vector)))

# getting the landscape section

skip_first_vals = 520  # this is probably 0 for the ResNets

shared_start = init1_vector[:skip_first_vals]

init1 = init1_vector[skip_first_vals:]
optimum1 = opt1_vector[skip_first_vals:]
optimum2 = opt2_vector[skip_first_vals:]

point_start = init1

# getting orthogonal bases
basis1 = optimum1 - init1
scale = np.linalg.norm(basis1)
basis1_normed = basis1 / np.linalg.norm(basis1)

basis2 = init2 - init1
basis2 = basis2 - basis1_normed * (np.sum(basis2 * basis1_normed))
basis2_normed = basis2 / np.linalg.norm(basis2)

# xs = np.linspace(-1.0,2.0,21)
# ys = np.linspace(-2.25,5.0,21)

# cut definitions
xs = np.linspace(-0.5, 2.0, 21)
ys = np.linspace(-0.5, 3.5, 21)

loss_grid = np.zeros((len(xs), len(ys))) * float("NaN")

for ix, x in enumerate(xs):
	for iy, y in enumerate(ys):
		params_now = point_start + scale * x * basis1_normed + scale * y * basis2_normed
		params_now = np.concatenate([shared_start, params_now], axis=0)

		# getting the new state dict using a fn I defined above
		new_dict = reform_params(keys_list, shapes_list, params_now)
		model.load_state_dict(new_dict)

		eval_dict_out_now = trainer.evaluate(eval_dataset=dataset_head1000)  # different for the ResNet for sure
		loss_now = eval_dict_out_now["eval_loss"]

		loss_grid[ix, iy] = loss_now

	print(ix, "/", len(xs))

	plt.contourf(xs, ys, loss_grid.T, cmap=plt.cm.Greens_r)
	plt.colorbar()
	plt.show()


# projecting points to the plane of visualization

def project_params(params_now):
	diff = params_now - point_start
	x_val = np.sum(diff * basis1_normed) / scale
	y_val = np.sum(diff * basis2_normed) / scale
	return x_val, y_val


plt.figure(figsize=(5.5, 5.0))

vmax = 10.0
vmin = 5.0

cmap = plt.cm.Greens_r

plt.title("Roberta on Esperanto Language Modelling", fontsize=16)
plt.contourf(xs, ys, loss_grid.T, cmap=cmap, levels=20, vmin=vmin, vmax=vmax)
plt.colorbar()

x_val, y_val = project_params(init1)
x_init1, y_init1 = x_val, y_val
plt.scatter([x_val], [y_val], marker="o", s=100, label="Init 1", color="crimson")

x_val, y_val = project_params(init2)
x_init2, y_init2 = x_val, y_val
plt.scatter([x_val], [y_val], marker="s", s=100, label="Init 2 (proj)", color="navy")

x_val, y_val = project_params(optimum1)
x_opt1, y_opt1 = x_val, y_val
plt.scatter([x_val], [y_val], marker="x", s=150, label="Opt 1", color="crimson")

x_val, y_val = project_params(optimum2)
x_opt2, y_opt2 = x_val, y_val
plt.scatter([x_val], [y_val], marker="x", s=150, label="Opt 2", color="navy")

plt.xlabel("Weight dir 1", fontsize=16)
plt.ylabel("Weight dir 2", fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16, framealpha=0.5)

plt.tight_layout()

plt.show()
