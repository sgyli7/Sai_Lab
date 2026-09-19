"""Startup isolation: failed launches must never control another listener."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import unittest

from sim2sim.godot_proc import GODOT_PROJECT,godot_bin,spawn_godot,stop_godot


@unittest.skipUnless(Path(godot_bin()).is_file() and
                     (GODOT_PROJECT/'generated/microduck/robot_spec.json').is_file(),
                     'Godot and generated robot resources are required')
class GodotStartupTests(unittest.TestCase):
    def test_occupied_port_is_rejected_without_connecting_to_its_owner(self):
        with socket.socket() as owner:
            owner.bind(('127.0.0.1',0));owner.listen();port=owner.getsockname()[1]
            with self.assertRaisesRegex(RuntimeError,'listen failed'):
                process,_,client=spawn_godot('res://main.tscn',port=port,recv_timeout=2.)
                stop_godot(process,client)
            owner.settimeout(.1)
            with self.assertRaises(TimeoutError):owner.accept()

    def test_concurrent_fresh_servers_complete_their_own_handshakes(self):
        def launch(_):
            process,port,client=spawn_godot('res://main.tscn',recv_timeout=3.)
            try:
                reply=client.call({'cmd':'hello'})
                self.assertTrue(reply['ok'])
                self.assertIsNone(process.poll())
                return process.pid
            finally:stop_godot(process,client)
        with ThreadPoolExecutor(max_workers=4) as pool:
            pids=list(pool.map(launch,range(8)))
        self.assertEqual(len(set(pids)),8)


if __name__=='__main__':unittest.main()
