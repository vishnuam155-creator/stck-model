import sys
import os
import csv
import subprocess
from datetime import datetime, timedelta
from decimal import Decimal
from django.core.management.base import BaseCommand
from django.utils import timezone
from signals.models import TradingSignal, ScanSession
from pathlib import Path


class Command(BaseCommand):
    help = 'Import trading signals from the intraday trading system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--capital',
            type=int,
            default=100000,
            help='Trading capital in rupees (default: 100000)'
        )
        parser.add_argument(
            '--stocks',
            type=int,
            default=50,
            help='Number of stocks to scan (default: 50)'
        )
        parser.add_argument(
            '--output',
            type=str,
            default='intraday_signals.csv',
            help='Output CSV filename (default: intraday_signals.csv)'
        )
        parser.add_argument(
            '--csv-only',
            action='store_true',
            help='Import from existing CSV file without running the scanner'
        )

    def handle(self, *args, **options):
        capital = options['capital']
        stocks = options['stocks']
        output_file = options['output']
        csv_only = options['csv_only']

        # Create a new scan session
        scan_session = ScanSession.objects.create(
            timestamp=timezone.now(),
            status='RUNNING'
        )

        start_time = timezone.now()

        try:
            if not csv_only:
                self.stdout.write(self.style.WARNING('Running intraday trading system...'))

                # Build the command to run the trading system
                base_dir = Path(__file__).resolve().parent.parent.parent.parent
                script_path = base_dir / 'intraday_trading_system.py'

                cmd = [
                    sys.executable,
                    str(script_path),
                    '--capital', str(capital),
                    '--stocks', str(stocks),
                    '--output', output_file
                ]

                # Run the trading system
                result = subprocess.run(
                    cmd,
                    cwd=str(base_dir),
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    self.stdout.write(self.style.ERROR(f'Error running trading system: {result.stderr}'))
                    scan_session.status = 'FAILED'
                    scan_session.save()
                    return

                self.stdout.write(self.style.SUCCESS('Trading system completed successfully!'))
            else:
                self.stdout.write(self.style.WARNING(f'Importing from existing CSV: {output_file}'))

            # Import the CSV file
            csv_path = Path(output_file)
            if not csv_path.exists():
                self.stdout.write(self.style.ERROR(f'CSV file not found: {output_file}'))
                scan_session.status = 'FAILED'
                scan_session.save()
                return

            signals_imported = self.import_csv(csv_path)

            # Update scan session
            end_time = timezone.now()
            scan_session.total_signals = signals_imported
            scan_session.buy_signals = TradingSignal.objects.filter(
                created_at__gte=start_time,
                signal_type='BUY'
            ).count()
            scan_session.sell_signals = TradingSignal.objects.filter(
                created_at__gte=start_time,
                signal_type='SELL'
            ).count()

            # Calculate total capital required and potential profit
            recent_signals = TradingSignal.objects.filter(created_at__gte=start_time)
            scan_session.total_capital_required = sum(
                signal.capital_required for signal in recent_signals
            )
            scan_session.total_potential_profit = sum(
                signal.reward_amount for signal in recent_signals
            )

            scan_session.scan_duration = end_time - start_time
            scan_session.status = 'COMPLETED'
            scan_session.save()

            self.stdout.write(
                self.style.SUCCESS(
                    f'\n✅ Successfully imported {signals_imported} signals!\n'
                    f'   Buy signals: {scan_session.buy_signals}\n'
                    f'   Sell signals: {scan_session.sell_signals}\n'
                    f'   Total capital required: ₹{scan_session.total_capital_required:,.2f}\n'
                    f'   Total potential profit: ₹{scan_session.total_potential_profit:,.2f}\n'
                )
            )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {str(e)}'))
            scan_session.status = 'FAILED'
            scan_session.save()
            raise

    def import_csv(self, csv_path):
        """Import signals from CSV file"""
        count = 0

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse the row data
                    signal_type = row.get('Signal', '').strip().upper()
                    if signal_type not in ['BUY', 'SELL']:
                        continue

                    # Map strategy names
                    strategy = row.get('Strategy', '').strip().upper()
                    if 'ORB' in strategy:
                        strategy = 'ORB'
                    elif 'VWAP' in strategy:
                        strategy = 'VWAP_PULLBACK'
                    elif 'CONFLUENCE' in strategy:
                        strategy = 'CONFLUENCE_REVERSAL'

                    # Create or update signal
                    signal, created = TradingSignal.objects.update_or_create(
                        symbol=row.get('Symbol', '').strip(),
                        timestamp__date=timezone.now().date(),
                        strategy=strategy,
                        defaults={
                            'signal_type': signal_type,
                            'entry_price': Decimal(row.get('Entry', '0').replace('₹', '').replace(',', '')),
                            'stop_loss': Decimal(row.get('Stop Loss', '0').replace('₹', '').replace(',', '')),
                            'target_price': Decimal(row.get('Target', '0').replace('₹', '').replace(',', '')),
                            'current_price': Decimal(row.get('Current', '0').replace('₹', '').replace(',', '')),
                            'rrr': Decimal(row.get('RRR', '0')),
                            'position_size': int(row.get('Position Size', '0')),
                            'capital_required': Decimal(row.get('Capital Required', '0').replace('₹', '').replace(',', '')),
                            'max_risk': Decimal(row.get('Risk', '0').replace('₹', '').replace(',', '')),
                            'risk_amount': Decimal(row.get('Risk', '0').replace('₹', '').replace(',', '')),
                            'reward_amount': Decimal(row.get('Reward', '0').replace('₹', '').replace(',', '')),
                            'confidence_score': Decimal(row.get('Confidence', '0').replace('%', '')),
                            'rsi': Decimal(row.get('RSI', '0')) if row.get('RSI') else None,
                            'vwap': Decimal(row.get('VWAP', '0').replace('₹', '').replace(',', '')) if row.get('VWAP') else None,
                            'atr': Decimal(row.get('ATR', '0')) if row.get('ATR') else None,
                            'volume_ratio': Decimal(row.get('Volume vs Avg', '0').replace('x', '')) if row.get('Volume vs Avg') else None,
                            'confluences': row.get('Confluences', ''),
                        }
                    )

                    if created:
                        count += 1

                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(f'Error importing row for {row.get("Symbol", "UNKNOWN")}: {str(e)}')
                    )
                    continue

        return count
