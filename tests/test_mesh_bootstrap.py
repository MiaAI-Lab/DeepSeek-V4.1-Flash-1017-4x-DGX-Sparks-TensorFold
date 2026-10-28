"""Offline bootstrap selection fixtures; no live NICs or NCCL required."""
import shlex
from test_scripts import ScriptTest


class MeshBootstrap(ScriptTest):
    def pairs(self, rows, count=4):
        fixture = self.sim / "inventory"
        fixture.write_text('\n'.join(rows) + '\n')
        code, out = self.bash('pair_links ' + ' '.join(map(str, range(count))) +
                              ' < ' + shlex.quote(str(fixture)))
        self.assertEqual(code, 0, out)
        ranks = {int(f[1]): f[2:] for line in out.splitlines()
                 if (f := line.split())[0] == 'rank'}
        return ranks, out

    def mesh(self, dual=False):
        rows = []
        for r in range(4):
            rows.append(f'{r} default management{r}')
            rails = [f'{r} fabric{r} hca{r} 192.0.2.{11+r}/24 3']
            if dual:
                rails.append(f'{r} second{r} other{r} 198.51.100.{11+r}/24 3')
                if r % 2 == 0:
                    rails.reverse()
            rows.extend(rails)
        return rows

    def test_shared_subnet_supplies_bootstrap(self):
        ranks, _ = self.pairs(self.mesh())
        self.assertEqual({r: fields[0] for r, fields in ranks.items()},
                         {r: f'fabric{r}' for r in range(4)})

    def test_dual_rail_choice_is_consistent_despite_inventory_order(self):
        ranks, _ = self.pairs(self.mesh(dual=True))
        for r, fields in ranks.items():
            self.assertEqual(fields[0], f'fabric{r}')
            expected = f'other{r},hca{r}' if r % 2 == 0 else f'hca{r},other{r}'
            self.assertEqual(fields[1:], [expected, '3'])  # Preserve data-plane HCA order/GID.

    def test_shared_subnet_does_not_require_a_default_route(self):
        ranks, _ = self.pairs([row for row in self.mesh() if ' default ' not in row])
        self.assertEqual({r: fields[0] for r, fields in ranks.items()},
                         {r: f'fabric{r}' for r in range(4)})

    def test_triangle_and_ring_retain_management_fallback(self):
        for count, edges in [(3, [(0, 1), (1, 2), (0, 2)]),
                             (4, [(0, 1), (1, 2), (2, 3), (3, 0)])]:
            with self.subTest(count=count):
                rows = [f'{r} default management{r}' for r in range(count)]
                for edge, (a, b) in enumerate(edges):
                    for r, suffix in [(a, 1), (b, 2)]:
                        rows.append(f'{r} nic{edge} hca{edge} 172.16.{edge}.{suffix}/24 3')
                ranks, out = self.pairs(rows, count)
                self.assertEqual({r: fields[0] for r, fields in ranks.items()},
                                 {r: f'management{r}' for r in range(count)})
                if count == 4:
                    self.assertIn('ring 0 1 2 3', out)

    def test_explicit_socket_override_wins_in_detect_links(self):
        for r in range(4):
            (self.sim / f'inventory{r}').write_text('\n'.join(
                row.split(' ', 1)[1] for row in self.mesh() if row.startswith(f'{r} ')) + '\n')
        code, out = self.bash(r'''
node_inventory() { cat "$SIM/inventory0"; }
worker() { cat "$SIM/inventory$1"; }
link_info() { echo '192.0.2.11 fabric0'; }
MASTER_ADDR=192.0.2.11
SOCKET_IFNAME=explicit0
detect_links
for r in 0 1 2 3; do echo "selected $r ${NODE_DEV[r]}"; done
''')
        self.assertEqual(code, 0, out)
        for r in range(4):
            self.assertIn(f'selected {r} explicit0', out)
