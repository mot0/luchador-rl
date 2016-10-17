from __future__ import division
from __future__ import absolute_import

import unittest

import luchador
from luchador.nn import (
    Tensor,
    Constant,
    Session,
    Adam,
    Adamax,
    scope as scp
)

'''
import logging
import theano
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'
logging.getLogger('luchador').setLevel(logging.DEBUG)
'''

BE = luchador.get_nn_backend()


def get_y_equals_x_squared(scope, x_init):
    with scp.variable_scope(scope):
        x = scp.get_variable(name='x', shape=(), trainable=True,
                             initializer=Constant(x_init))
        y = Tensor(x.unwrap() * x.unwrap(), shape=())
    return x, y


def get_slot_var(optimizer, slot_name, var_name=None):
    opt_name = optimizer.args['name']
    names = ['{}/{}'.format(opt_name, slot_name)]
    if var_name:
        names.append('{}/{}/{}'.format(opt_name, var_name, slot_name))

    for v in optimizer.get_parameter_variables():
        if v.name in names:
            return v
    raise ValueError('No slot was found')


class AdamTest(unittest.TestCase):
    def test_beta_power_update(self):
        """Beta paramete is updated every time update is evaluated"""
        beta1, beta2, x_init_val, name = 0.9, 0.999, 3.0, 'Adam'

        adam = Adam(learning_rate=0.01, beta1=beta1, beta2=beta2, name=name)

        x_tensor, y_tensor = get_y_equals_x_squared(
            scope=self.id().replace('.', '/'), x_init=x_init_val)

        minimize_op = adam.minimize(loss=y_tensor, wrt=x_tensor)
        beta1_pow_tensor = get_slot_var(adam, 'beta1_power')
        beta2_pow_tensor = get_slot_var(adam, 'beta2_power')
        m_tensor = get_slot_var(adam, 'm', var_name=x_tensor.name)
        v_tensor = get_slot_var(adam, 'v', var_name=x_tensor.name)

        session = Session()
        session.initialize()

        x_val_prev, m_val_prev, v_val_prev = x_init_val, 0, 0
        for i in range(1, 10):
            session.run(updates=minimize_op, name='optimization')

            beta1_pow_val, beta2_pow_val = session.run(
                outputs=[beta1_pow_tensor, beta2_pow_tensor],
                name='fetch1',
            )

            expected = beta1 ** (i + 1)
            found = beta1_pow_val
            diff = abs(expected - found)
            self.assertTrue(diff < 0.01,
                            'Beta1 is not correctly updated. '
                            'Expected: {}, Found: {}'.format(expected, found))

            expected = beta2 ** (i + 1)
            found = beta2_pow_val
            diff = abs(expected - found)
            self.assertTrue(diff < 0.01,
                            'Beta2 is not correctly updated. '
                            'Expected: {}, Found: {}'.format(expected, found))

            x_val = session.run(outputs=x_tensor, name='fetch2')
            self.assertTrue(
                0 <= x_val < x_val_prev,
                'The value of `x` must regress to zero at each update. '
                'Previous value: {}, current value: {}'
                .format(x_val_prev, x_val)
            )
            x_val_prev = x_val

            m_val, v_val = session.run(
                outputs=[m_tensor, v_tensor], name='fetch3')
            grad = 2 * x_val_prev
            expected = m_val_prev + (1.0 - beta1) * (grad - m_val_prev)
            found = m_val
            diff = abs(expected - found)
            self.assertTrue(
                diff < 0.01,
                'The value of `m` is not correctly updated. '
                'Expected: {}, Fround: {}'
                .format(expected, found)
            )
            m_val_prev = m_val

            expected = v_val_prev + (1.0 - beta2) * (grad*grad - v_val_prev)
            found = v_val
            diff = abs(expected - found)
            self.assertTrue(
                diff < 0.01,
                'The value of `v` is not correctly updated. '
                'Expected: {}, Fround: {}'
                .format(expected, found)
            )
            v_val_prev = v_val


class AdamaxTest(unittest.TestCase):
    def test_beta_power_update(self):
        """Beta parameter is updated every time update is evaluated"""
        beta1, beta2, x_init_val = 0.9, 0.999, 3.0
        adamax = Adamax(learning_rate=0.01, beta1=beta1)

        x_tensor, y_tensor = get_y_equals_x_squared(
            scope=self.id().replace('.', '/'), x_init=x_init_val)

        minimize_op = adamax.minimize(loss=y_tensor, wrt=x_tensor)
        beta1_pow_tensor = get_slot_var(adamax, 'beta1_power')
        m_tensor = get_slot_var(adamax, 'm', var_name=x_tensor.name)
        u_tensor = get_slot_var(adamax, 'u', var_name=x_tensor.name)

        session = Session()
        session.initialize()

        x_val_prev, m_val_prev, u_val_prev = x_init_val, 0, 0
        for i in range(1, 10):
            session.run(updates=minimize_op, name='optimization')

            beta1_pow_val = session.run(
                outputs=beta1_pow_tensor, name='fetch1')

            expected = beta1 ** (i + 1)
            found = beta1_pow_val
            diff = abs(expected - found)
            self.assertTrue(diff < 0.01,
                            'Beta1 is not correctly updated. '
                            'Expected: {}, Found: {}'.format(expected, found))

            m_val, u_val = session.run(
                outputs=[m_tensor, u_tensor], name='fetch3')
            grad = 2 * x_val_prev
            expected = m_val_prev + (1.0 - beta1) * (grad - m_val_prev)
            found = m_val
            diff = abs(expected - found)
            self.assertTrue(
                diff < 0.01,
                'The value of `m` is not correctly updated. '
                'Expected: {}, Fround: {}'
                .format(expected, found)
            )
            m_val_prev = m_val

            expected = max(u_val_prev * beta2, abs(grad))
            found = u_val
            diff = abs(expected - found)
            self.assertTrue(
                diff < 0.01,
                'The value of `u` is not correctly updated. '
                'Expected: {}, Fround: {}'
                .format(expected, found)
            )
            u_val_prev = u_val

            x_val = session.run(outputs=x_tensor, name='fetch2')
            self.assertTrue(
                0 <= x_val < x_val_prev,
                'The value of `x` must regress to zero at each update. '
                'Previous value: {}, current value: {}'
                .format(x_val_prev, x_val)
            )
            x_val_prev = x_val