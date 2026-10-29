"""Exercise preparation utility commands with only the runtime image available."""
from test_scripts import BASH, ScriptTest


class PrepareImage(ScriptTest):
    def test_both_kit_checks_use_runtime_image(self):
        kit = self.tmp / 'kit'
        (kit / 'kit').mkdir(parents=True)
        verify = kit / 'kit/verify_kit.sh'
        verify.write_text('#!' + BASH + r'''
echo "$IMAGE" >> "$SIM/kit-images"
[[ "$IMAGE" == runtime-only ]] || exit 42
if [[ "${BIAS_DIFF:-0}" == 1 ]]; then
  echo 'DIFFERS (bias): vision/gate_bias_vl.safetensors'
  echo '355 of 356 files match; 1 differ, 0 missing'
else
  echo '356 of 356 files match'
fi
''')
        verify.chmod(0o755)
        source = (self.repo / 'scripts/prepare.sh').read_text()
        checks = source[source.index('kit_out='):source.index('log "Kit:')]
        for bias in ('0', '1'):
            with self.subTest(bias=bias):
                record = self.sim / 'kit-images'
                record.unlink(missing_ok=True)
                code, out = self.bash('KIT_DIR=' + str(kit) + '\nSNAP=unused\n'
                                      'die() { echo "$*" >&2; exit 1; }\n' + checks,
                                      {'IMAGE': 'runtime-only', 'BASE_IMAGE': 'base-missing', 'BIAS_DIFF': bias})
                self.assertEqual(code, 0, out)
                self.assertEqual(record.read_text().splitlines(), ['runtime-only', 'runtime-only'])

    def test_worker_inventory_and_copy_need_no_base_image(self):
        # Fail if Docker would need the absent base image; never invoke a real daemon.
        docker = self.tmp / 'bin/docker'
        docker.write_text('#!' + BASH + r'''
printf '%s\n' "$@" >> "$SIM/utility-args"
[[ " $* " != *' base-missing '* ]] || exit 42
[[ " $* " == *' runtime-only '* ]] || exit 43
echo 'fixture 1'
''')
        docker.chmod(0o755)
        lines = (self.repo / 'scripts/prepare.sh').read_text().splitlines()
        inventory = next(l for l in lines if l.lstrip().startswith('have=$(worker ') and '--entrypoint find' in l)
        copy = next(l for l in lines if l.lstrip().startswith('worker ') and '--entrypoint bash' in l)
        setup = '''
i=1
NFS_VOLUME=fixture
DATA_SUB=data
KIT_SUB=kit
SNAP_REL=snapshot
WORKER_ENGRAM_DIR=/fixture-out
shards=(table.dat)
snapshot_rev() { echo fixture; }
'''
        for name, command in [('inventory', inventory + '\n[[ "$have" == "fixture 1" ]]'),
                              ('copy', copy.rstrip().removesuffix(' ||'))]:
            with self.subTest(command=name):
                record = self.sim / 'utility-args'
                record.unlink(missing_ok=True)
                code, out = self.bash(setup + command + '\n',
                                      {'IMAGE': 'runtime-only', 'BASE_IMAGE': 'base-missing'})
                self.assertEqual(code, 0, out)
                self.assertIn('runtime-only', record.read_text().splitlines())
                self.assertNotIn('base-missing', record.read_text().splitlines())
