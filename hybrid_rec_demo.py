from flask import Flask, render_template, request, session
from recommender_system.create_clusters import build_clusters
from recommender_system.util import load_data, expert_base
from recommender_system.batch_MF import train
from recommender_system.user_rec import user_rec

app = Flask(__name__)
app.secret_key = '\xd5\\x\xf6h6\xe1\x1f\xf3\xb9\x91\xa7\x93\x1a\xcd\xe9\xc4\\\xbd7\xea\xf32\x13'

clusters = []
clusters_index = []
experts_index = []
users_index = []
v_batch = []
bias = 0


@app.route('/', methods=['GET'])
@app.route('/choose_cluster', methods=['GET'])
def choose_clusters():
    return render_template("cluster_choice.html", cluster_num=len(clusters))


@app.route('/choose_user', methods=['POST'])
def choose_users():
    n_cluster = int(request.form["n_cluster"])
    session['n_cluster'] = n_cluster
    users = clusters_index[n_cluster]
    return render_template("user_choice.html", users=users)


@app.route('/show_rec', methods=['POST'])
def show_recommendation():
    n_user = int(request.form["n_user"])
    n_cluster = int(session['n_cluster'])
    session['n_user'] = int(clusters_index[n_cluster][n_user])
    results = user_rec(n_user, clusters[n_cluster], v_batch, bias)

    return render_template("show_rec.html", results = results)


@app.context_processor
def inject_enumerate():
    return dict(enumerate=enumerate)

if __name__ == '__main__':
    data = load_data()
    experts_index, users_index = expert_base(data)
    users_matrix = data[users_index]
    experts_matrix = data[experts_index]
    clusters, clusters_index = build_clusters(users_matrix, n_cluster=10)
    N = len(experts_matrix)
    M = len(experts_matrix[0])
    K = 40
    u_batch, v_batch, bias = train(experts_matrix, N, M, K, suffix_name='experts')
    app.run(debug=True)
