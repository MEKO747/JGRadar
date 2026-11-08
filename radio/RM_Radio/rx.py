#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless RX2 flowgraph adapted from T-DT's 2026 red/blue receivers."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 2_000_000


def load_settings(path, team, noise_cutoff=None):
    config = json.loads(Path(path).read_text())
    if team not in ('red', 'blue'):
        raise ValueError('team must be red or blue')
    profile = config['teams'][team]
    gain = float(config['gain'])
    if not 0 <= gain <= 60 or config['gain_mode'] not in ('manual', 'slow_attack', 'fast_attack'):
        raise ValueError('invalid RX gain or gain mode')
    if not isinstance(config['uri'], str) or not config['uri'].startswith('ip:') or not config['expected_serial']:
        raise ValueError('IP URI and expected RX serial are required')
    if noise_cutoff is not None and not 100000 <= noise_cutoff <= 480000:
        raise ValueError('noise cutoff must be 100000..480000 Hz')
    lo = int(profile['lo_frequency'])
    if not 325000000 <= lo <= 3800000000 or len(profile['noise_offsets']) != 2:
        raise ValueError('invalid LO frequency or noise channel configuration')
    channels = [dict(name='info', offset=int(profile['info_offset']), cutoff=270000,
                     demod_cutoff=250000, transition=50000, port=5555)]
    for index, offset in enumerate(profile['noise_offsets']):
        channels.append(dict(name=f'noise{index + 1}', offset=int(offset),
                             cutoff=noise_cutoff or (480000, 420000)[index],
                             demod_cutoff=(480000, 410000)[index], transition=40000,
                             port=6666 + index))
    for channel in channels:
        if abs(channel['offset']) + channel['cutoff'] + 20000 >= SAMPLE_RATE / 2:
            raise ValueError('channel exceeds the RX sampling band')
        channel['frequency'] = lo + channel['offset']
    return dict(uri=config['uri'], expected_serial=str(config['expected_serial']), gain=gain,
                gain_mode=config['gain_mode'], team=team, lo_frequency=lo,
                sample_rate=SAMPLE_RATE, sps=47, input='RX2', channels=channels)


def load_decoder_module():
    import gnuradio
    package_path = ROOT / 'install/oot/lib/python3/dist-packages/gnuradio'
    if not package_path.is_dir():
        raise RuntimeError('local decoder is not built; run bash build.sh first')
    # Prioritize this delivery's decoder over any older system OOT installation.
    gnuradio.__path__.insert(0, str(package_path))
    from gnuradio import robomaster
    if not Path(robomaster.__file__).resolve().is_relative_to(package_path):
        raise RuntimeError('another robomaster module was imported before this delivery')
    return robomaster


def build_flowgraph(settings, source=None):
    robomaster = load_decoder_module()
    from gnuradio import analog, filter, gr, iio, zeromq
    from gnuradio.filter import firdes
    from gnuradio.fft import window
    tb = gr.top_block('T-DT RX2 → ROS 2', catch_exceptions=True)
    if source is None:
        identity = subprocess.run(['iio_attr', '-u', settings['uri'], '-T', '5000', '-q', '-C', 'hw_serial'],
                                  check=True, capture_output=True, text=True, timeout=8)
        serial = identity.stdout.strip().split(':')[-1].strip()
        if serial != settings['expected_serial']:
            raise RuntimeError(f'RX serial mismatch: expected {settings["expected_serial"]}, got {serial}')
        source = iio.fmcomms2_source_fc32(settings['uri'], [False, False, True, True], 65536)
        source.set_frequency(settings['lo_frequency'])
        source.set_samplerate(SAMPLE_RATE)
        source.set_gain_mode(1, settings['gain_mode'])
        if settings['gain_mode'] == 'manual':
            source.set_gain(1, settings['gain'])
        source.set_quadrature(True)
        source.set_rfdc(True)
        source.set_bbdc(True)
        source.set_filter_params('Auto', '', 0, 0)
    for channel in settings['channels']:
        channel_filter = filter.freq_xlating_fir_filter_ccc(
            2, firdes.low_pass(1.0, SAMPLE_RATE, channel['cutoff'], 40000), channel['offset'], SAMPLE_RATE)
        demod = analog.quadrature_demod_cf(1.5)
        lowpass = filter.fir_filter_fff(1, firdes.low_pass(
            1, SAMPLE_RATE / 2, channel['demod_cutoff'], channel['transition'], window.WIN_BLACKMAN, 6.76))
        decoder = (robomaster.gfsk_receiver(1100, 47, 27) if channel['name'] == 'info' else
                   robomaster.gfsk_receiver_noise(900, 47, 27))
        publisher = zeromq.pub_sink(gr.sizeof_char, 1, f'tcp://127.0.0.1:{channel["port"]}',
                                    100, False, -1, '', True, True)
        tb.connect(source, channel_filter, demod, lowpass, decoder, publisher)
    return tb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--team', required=True, choices=('red', 'blue'), help='own team color')
    parser.add_argument('--config', type=Path, default=ROOT / 'rx.json')
    parser.add_argument('--noise-cutoff', type=int, help='lab override only; original field filters are default')
    parser.add_argument('--describe', action='store_true', help='show configuration without accessing hardware')
    parser.add_argument('--seconds', type=float, help='optional finite RX connection check')
    args = parser.parse_args()
    if args.seconds is not None and not 0 < args.seconds <= 300:
        parser.error('seconds must be (0,300]')
    settings = load_settings(args.config, args.team, args.noise_cutoff)
    print(json.dumps(settings, ensure_ascii=False), flush=True)
    if args.describe:
        return
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    tb = build_flowgraph(settings)
    try:
        tb.start()
        stop.wait(args.seconds)
    finally:
        tb.stop()
        tb.wait()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'RX failed: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
