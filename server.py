"""
Servidor Flask para conectar frontend com backend
Expõe API REST para executar simulações
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime

# Importar classes do sistema principal
from simulation import Simulation, run_baseline, run_experiment

app = Flask(__name__, static_folder='static')
CORS(app)

# Diretório para salvar resultados
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


@app.route('/')
def index():
    """Serve a página principal"""
    return send_from_directory('static', 'index.html')


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Endpoint para executar uma simulação
    
    Body esperado:
    {
        "approach": "A" | "B" | "C",
        "groupType": "homogeneous" | "heterogeneous" | "baseline",
        "numAgents": 2-10,
        "bombRatio": 0.5-0.8
    }
    """
    try:
        data = request.json
        
        approach = data.get('approach', 'A')
        group_type = data.get('groupType', 'homogeneous')
        num_agents = int(data.get('numAgents', 3))
        bomb_ratio = float(data.get('bombRatio', 0.5))
        
        # Validações
        if approach not in ['A', 'B', 'C']:
            return jsonify({'error': 'Abordagem inválida'}), 400
        
        if group_type not in ['homogeneous', 'heterogeneous', 'baseline']:
            return jsonify({'error': 'Tipo de grupo inválido'}), 400
        
        if not (2 <= num_agents <= 10):
            return jsonify({'error': 'Número de agentes deve estar entre 2 e 10'}), 400
        
        # Executar simulação
        if group_type == 'baseline':
            metrics = run_baseline(approach, bomb_ratio)
            environment_states = None  # Baseline não retorna estado do ambiente
        else:
            sim = Simulation(approach, num_agents, bomb_ratio, group_type)
            sim.collect_states = True  # Ativar coleta de estados para animação
            
            metrics = sim.run()
            
            # Usar estados coletados durante a simulação
            environment_states = sim.states
        
        # Salvar resultado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"{RESULTS_DIR}/sim_{approach}_{group_type}_{timestamp}.json"
        
        result_data = {
            'config': data,
            'metrics': metrics,
            'environment_states': environment_states,
            'timestamp': timestamp
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'environment_states': environment_states,
            'saved_to': result_file
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/experiment', methods=['POST'])
def run_experiment_endpoint():
    """
    Endpoint para executar experimento completo (30 repetições)
    
    Body esperado:
    {
        "approach": "A" | "B" | "C",
        "groupType": "homogeneous" | "heterogeneous",
        "numAgents": 2-10,
        "bombRatio": 0.5-0.8,
        "repetitions": 30
    }
    """
    try:
        data = request.json
        
        approach = data.get('approach', 'A')
        group_type = data.get('groupType', 'homogeneous')
        num_agents = int(data.get('numAgents', 3))
        bomb_ratio = float(data.get('bombRatio', 0.5))
        repetitions = int(data.get('repetitions', 30))
        
        # Executar experimento
        results_df = run_experiment(approach, num_agents, bomb_ratio, group_type, repetitions)
        
        # Calcular estatísticas
        stats = {
            'mean_execution_time': float(results_df['execution_time'].mean()),
            'std_execution_time': float(results_df['execution_time'].std()),
            'min_execution_time': float(results_df['execution_time'].min()),
            'max_execution_time': float(results_df['execution_time'].max()),
            'success_rate': float(results_df['success'].mean() * 100),
            'mean_steps': float(results_df['total_steps'].mean()),
            'mean_exploration': float(results_df['exploration_percentage'].mean())
        }
        
        # Salvar CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"{RESULTS_DIR}/experiment_{approach}_{group_type}_{num_agents}agents_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)
        
        return jsonify({
            'success': True,
            'statistics': stats,
            'csv_file': csv_file,
            'repetitions': repetitions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/results', methods=['GET'])
def list_results():
    """Lista todos os resultados salvos"""
    try:
        files = []
        for filename in os.listdir(RESULTS_DIR):
            if filename.endswith('.json') or filename.endswith('.csv'):
                filepath = os.path.join(RESULTS_DIR, filename)
                files.append({
                    'filename': filename,
                    'size': os.path.getsize(filepath),
                    'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                })
        
        return jsonify({
            'success': True,
            'files': files,
            'total': len(files)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/result/<filename>', methods=['GET'])
def get_result(filename):
    """Retorna um resultado específico"""
    try:
        filepath = os.path.join(RESULTS_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Arquivo não encontrado'}), 404
        
        if filename.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        
        elif filename.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(filepath)
            return jsonify(df.to_dict('records'))
        
        else:
            return jsonify({'error': 'Formato não suportado'}), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/compare', methods=['POST'])
def compare_results():
    """
    Compara resultados de diferentes configurações
    
    Body esperado:
    {
        "approach": "A",
        "bombRatio": 0.5,
        "numAgents": 3
    }
    
    Retorna comparação entre homogêneo, heterogêneo e baseline
    """
    try:
        data = request.json
        approach = data.get('approach', 'A')
        bomb_ratio = float(data.get('bombRatio', 0.5))
        num_agents = int(data.get('numAgents', 3))
        
        # Executar as três configurações
        results = {}
        
        for group_type in ['homogeneous', 'heterogeneous']:
            results_df = run_experiment(approach, num_agents, bomb_ratio, group_type, repetitions=30)
            
            results[group_type] = {
                'success_rate': float(results_df['success'].mean() * 100),
                'mean_time': float(results_df['execution_time'].mean()),
                'std_time': float(results_df['execution_time'].std()),
                'mean_steps': float(results_df['total_steps'].mean())
            }
        
        # Baseline
        baseline_results = []
        for i in range(30):
            metrics = run_baseline(approach, bomb_ratio, seed=i)
            baseline_results.append(metrics)
        
        import pandas as pd
        baseline_df = pd.DataFrame(baseline_results)
        
        results['baseline'] = {
            'mean_time': float(baseline_df['execution_time'].mean()),
            'std_time': float(baseline_df['execution_time'].std()),
            'mean_steps': float(baseline_df['steps'].mean()) if 'steps' in baseline_df else 0
        }
        
        return jsonify({
            'success': True,
            'comparison': results,
            'config': {
                'approach': approach,
                'bombRatio': bomb_ratio,
                'numAgents': num_agents
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Verifica se o servidor está funcionando"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        SERVIDOR FLASK - SISTEMA DE AGENTES INTELIGENTES     ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Servidor iniciado em: http://localhost:5000
    
    Endpoints disponíveis:
      GET  /                     - Interface web
      POST /api/simulate         - Executar uma simulação
      POST /api/experiment       - Executar 30 repetições
      POST /api/compare          - Comparar configurações
      GET  /api/results          - Listar resultados
      GET  /api/result/<file>    - Obter resultado específico
      GET  /api/health           - Health check
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
