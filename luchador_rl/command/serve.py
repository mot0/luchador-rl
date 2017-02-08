"""Module to define ``luchador serve`` subcommand"""
from __future__ import absolute_import

import logging

from paste.translogger import TransLogger

from luchador import nn
from luchador.util import load_config
from luchador_rl.util import create_server
import luchador_rl.env.remote
import luchador_rl.agent.remote

_LG = logging.getLogger(__name__)


def _run_server(app, port):
    server = create_server(TransLogger(app), port=port)
    app.attr['server'] = server
    _LG.info('Starting server on port %d', port)
    try:
        server.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        _LG.info('Server on port %d stopped.', port)


###############################################################################
def entry_point_env(args):
    """Entry point for ``luchador serve env`` command"""
    if args.environment is None:
        raise ValueError('Environment config is not given')
    env_config = load_config(args.environment)
    env = luchador_rl.env.get_env(env_config['typename'])(**env_config['args'])
    app = luchador_rl.env.remote.create_env_app(env)
    _run_server(app, args.port)


def entry_point_manager(args):
    """Entry point for ``luchador serve manager`` command"""
    app = luchador_rl.env.remote.create_manager_app()
    _run_server(app, args.port)


def entry_point_param(args):
    """Entry point for ``luchador serve parameter``"""
    nn.make_model(args.model)
    session = nn.Session()
    session.initialize()
    app = luchador_rl.agent.remote.create_parameter_server_app(session)
    _run_server(app, args.port)
